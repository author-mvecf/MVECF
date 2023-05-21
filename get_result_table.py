import numpy as np
from lib.utils import get_mean_variance, get_data, get_ret_data, load_pickle
from lib.analysis_utils import get_model, get_name
import os
import pandas as pd
from sklearn import metrics
import sys


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def rec_analysis(total_score, holdings_data, topk=20):
    test_data = holdings_data["test_data"]
    indptr = holdings_data["test_indptr"]
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]
    y_true = test_data[2]

    y_score = total_score[test_data[0], test_data[1]]
    output = {}
    result_recall = 0
    result_precision = 0
    result_accuracy = 0
    result_map_at_k = 0
    map = 0
    mrr = 0
    for user in range(m):
        tmp_true = y_true[indptr[user]:indptr[user + 1]]
        total_num_relevant = tmp_true.sum()
        tmp_score = y_score[indptr[user]:indptr[user + 1]]
        sorted_index = tmp_score.argsort()[::-1]
        tmp_true = tmp_true[sorted_index]
        tmp_score = tmp_score[sorted_index]

        first_relavant_loc = tmp_true.argmax() + 1
        mrr += 1 / first_relavant_loc / m
        map += metrics.average_precision_score(tmp_true, tmp_score) / m

        TN = (1 - tmp_true[topk:]).sum()

        tmp_true = tmp_true[:topk]
        # tmp_score = tmp_score[:k]
        # tmp_pred = np.ones(len(tmp_score))
        TP = sum(tmp_true)
        if total_num_relevant == 0:
            recall = 0
        else:
            recall = TP / min(total_num_relevant, topk)
        precision = TP / topk

        ap_at_k = 0
        for i in (np.where(tmp_true == 1)[0]):
            ap_at_k += tmp_true[:i + 1].sum() / min(total_num_relevant, i + 1) / topk

        result_recall += recall / m
        result_precision += precision / m
        result_accuracy += (TP + TN) / n / m
        result_map_at_k += ap_at_k / m
    output["recall@{}".format(topk)] = result_recall
    output["precision@{}".format(topk)] = result_precision
    output["accuracy@{}".format(topk)] = result_accuracy
    output["MAP@{}".format(topk)] = result_map_at_k
    output["MRR"] = mrr
    output["MAP"] = map
    return output


def get_real_mv_user(items, ret_data_all, rebalance=False):
    ret = ret_data_all[items]

    if not rebalance:
        wealth_stock = np.cumprod(1 + ret, axis=1)
        wealth = wealth_stock.mean(axis=0)
        wealth = np.append([1], wealth)
        ret_u = (wealth[1:] / wealth[:-1]) - 1
    else:
        ret_u = np.mean(ret, axis=0)

    mu_u = ret_u.mean() * 52
    sig_u = ret_u.std() * (52 ** 0.5)

    return mu_u, sig_u


def get_sr_statistics(
        mv_list, mv_train_list, mv_insample_list, mv_insample_train_list,
):
    def get_output(mv, mv_train, is_insample=False):
        mean_list = np.array(mv).T[0]
        risk_list = np.array(mv).T[1]
        sr_list = mean_list / risk_list

        mean_train = np.array(mv_train).T[0]
        risk_train = np.array(mv_train).T[1]
        sr_train = mean_train / risk_train

        sr_diff_train = sr_list - sr_train
        mean_diff_train = mean_list - mean_train
        risk_diff_train = risk_list - risk_train

        output = {
            "delta_sr": sr_diff_train.mean(),
            "delta_mean": mean_diff_train.mean(),
            "delta_risk": risk_diff_train.mean(),
            "prob_delta_sr_positive": (sr_diff_train > 0).sum() / len(sr_list),
        }
        output = pd.DataFrame.from_dict(output, orient="index")
        if not is_insample:
            output.index = output.index + "_expost"
        return output

    output_backtest = get_output(mv_list, mv_train_list, is_insample=False)
    output_insample = get_output(mv_insample_list, mv_insample_train_list, is_insample=True)
    output = pd.concat([output_backtest, output_insample])
    return output[0].to_dict()


def mv_to_sr(mv_list):
    mv_list = np.array(mv_list).T
    sr_list = mv_list[0] / mv_list[1]
    return sr_list


def calc_sr_models(total_score, holdings_data, ret_data, mu, cov, topk=20, rebalance=False):
    train_data = holdings_data["train_data"]
    train_indptr = holdings_data["train_indptr"]
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]

    mv = []
    best_item = []
    recommended_all = np.zeros((m, n))
    for user in range(m):
        train_items = train_data[1][train_indptr[user]:train_indptr[user + 1]]
        if total_score is None:
            recommended_port = train_items
        else:
            candidate_items = list(set(range(n)) - set(train_items))
            candidate_items.sort()
            candidate_items = np.array(candidate_items)
            tmp_score = total_score[user][candidate_items]
            sorted_index = tmp_score.argsort()[::-1]  # top k sorted
            sorted_items = candidate_items[sorted_index]
            # sorted_scores = tmp_score[sorted_index] # 이건 threshold 사용할 경우 필요

            best_item.append(sorted_items[0])
            recommended_port = np.append(train_items, sorted_items[:topk])
        recommended_all[user, recommended_port] = 1 / len(recommended_port)
        mean, risk = get_real_mv_user(recommended_port, ret_data, rebalance=rebalance)

        mv.append([mean, risk])
    mean_insample = np.matmul(mu, recommended_all.T)
    risk_insample = np.sqrt(np.diag(np.matmul(np.matmul(recommended_all, cov), recommended_all.T)))
    return mv, np.c_[mean_insample, risk_insample]


if __name__ == "__main__":
    data_type = "CRSP"
    target_year = 2012

    sr_performance = {}
    rec_performance = {}
    analysis_path = os.path.join("results", "analysis.xlsx")
    index_name = ["data_type", "year", "model", "lr", "reg_param", "n_factors", "gamma", "reg_param_mv"]

    save_path = "results"
    ex_post_test_years = 5
    topk = 20
    model_list = [
        "wmf",
        "bpr_nov",
        "mvecf_reg",
        "mvecf_wmf",
        "mvecf_lgcn",
        "twophase_wmf"
    ]
    # load main data
    holdings_data, factor_params = get_data(data_type, target_year)
    mu, cov = get_mean_variance(factor_params)
    m = holdings_data["n_users"]
    n = holdings_data["n_items"]

    # load return data
    ret_data = get_ret_data(data_type, target_year)
    ret_data = ret_data[
        (ret_data.index.year > target_year) & (ret_data.index.year <= target_year + ex_post_test_years)
        ]
    ret_data = ret_data.fillna(0)
    ret_data = ret_data.T.values
    assert len(ret_data) == n

    mv_train, mv_train_insample = calc_sr_models(None, holdings_data, ret_data, mu, cov)
    sr_train = mv_to_sr(mv_train)

    for model_type in model_list:
        print(model_type)
        dirpath = os.path.join(save_path, model_type)
        if "twophase" in model_type:
            total_score = load_pickle(os.path.join(dirpath, "total_score.pkl"))
        else:
            model = get_model(dirpath)
            if model is None:
                continue
            assert m == model.n_users
            assert n == model.n_items

            total_score = []
            for user in range(m):
                total_score.append(
                    model.forward(user, np.arange(model.n_items))
                )
            total_score = np.array(total_score)

        name = get_name(model, data_type, target_year, model_type, index_name)
        mv_model, mv_model_insample = calc_sr_models(
            total_score, holdings_data, ret_data, mu, cov, topk=topk)

        sr_performance[name] = get_sr_statistics(mv_model, mv_train, mv_model_insample, mv_train_insample)
        rec_performance[name] = rec_analysis(total_score, holdings_data, topk=topk)

    sr_results_table = pd.DataFrame.from_dict(sr_performance, orient="index")
    rec_results_table = pd.DataFrame.from_dict(rec_performance, orient="index")

    data = pd.concat([sr_results_table, rec_results_table], axis=1)
    data.index.set_names(index_name, inplace=True)
    data.to_excel(analysis_path)

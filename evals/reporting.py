
def cal_metrics(rows:list[dict])->dict:

    individual_metrics={}
    if not rows:
        return individual_metrics
    
    for row in rows:
        dm = row.get("deepeval_metrics")
        if not dm:
            continue
        individual_metrics[row["id"]] = {
            m["name"]: {"name": m["name"], "score": m.get("score"), "reason": m.get("reason")}
            for m in dm if m.get("name")
        }

    return individual_metrics

def aggregate(metrics_by_id:dict[str,list[dict]])->dict:
    
    agg_metrics={}
    if not metrics_by_id:
        return agg_metrics

    for metrics_list in metrics_by_id.values():
        if not metrics_list:
            continue
        for metric in metrics_list:
            name = metric.get("name")
            if name is None:
                continue
            if name not in agg_metrics:
                agg_metrics[name]={"total":0,"passes":0,"scores_sum":0,"scores_count":0}
            agg_metrics[name]["total"] += 1
            if metric.get("success"):
                agg_metrics[name]["passes"] += 1
            if metric.get("score") is not None:
                agg_metrics[name]["scores_sum"] += metric["score"]
                agg_metrics[name]["scores_count"] += 1

    return agg_metrics

def print_table(rows:list[dict]):

    individual_metrics = cal_metrics(rows)
    metrics_by_id = {
        row["id"]: row["deepeval_metrics"]
        for row in rows if row.get("deepeval_metrics")
    }

    agg_metrics = aggregate(metrics_by_id)

    print("==================RESULTS=============================")
    print()
    print("| TESTCASE | FAITH | ANS_REL | CONTEXT_PREC | CONTEXT_RECALL | CONTEXT_REL |")
    for testcase, metrics in individual_metrics.items():
        print(f'| {testcase} |', end="")
        for val in metrics.values():
            score = val.get("score")
            print(f' {score if score is not None else "ERR"} |', end="")
        print()
 
    print("-------------------AGGREGATE RESULTS-----------------------")
    print("| METRIC | AVERAGE_SCORE | PASS_RATE | TOTAL |")
    for metric_name, agg_metric in agg_metrics.items():
        scores_count = agg_metric["scores_count"]
        total = agg_metric["total"]
        avg_score = agg_metric["scores_sum"] / scores_count if scores_count > 0 else None
        pass_rate = (agg_metric["passes"] / total) * 100 if total > 0 else None
 
        avg_str = f"{avg_score:.2f}" if avg_score is not None else "N/A"
        pass_str = f"{pass_rate:.2f}%" if pass_rate is not None else "N/A"
        print(f"| {metric_name} | {avg_str} | {pass_str} | {total} |")
 

# if __name__ == "__main__":
    # test_results = [TestResult(name="test1",
    #                            metrics_data=[
    #                                 MetricData(name="Faithfulness",threshold=2,success=True,score=56,reason="fdvj"),
    #                                 MetricData(name="ans",threshold=2,success=True,score=96,reason="fdvj"),
    #                                 MetricData(name="cts",threshold=2,success=True,score=86,reason="fdvj"),
    #                                 MetricData(name="ctx rel",threshold=2,success=True,score=46,reason="fdvj")
    #                             ],
    #                             success=True,
    #                             conversational=True),
    #     TestResult(name="test2",metrics_data=[
    #     MetricData(name="Faithfulness",threshold=2,success=True,score=56,reason="fdvj"),
    #     MetricData(name="ans",threshold=2,success=True,score=96,reason="fdvj"),
    #     MetricData(name="cts",threshold=2,success=True,score=86,reason="fdvj"),
    #     MetricData(name="ctx rel",threshold=2,success=True,score=46,reason="fdvj")],success=True,
    #                                     conversational=True)]
    # print(print_table(test_results))
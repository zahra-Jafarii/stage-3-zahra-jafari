SELECT
    id,
    target,
    LEAST(revolving_utilization, {util_cap}) AS revolving_utilization,
    age,
    CASE WHEN past_due_30_59 IN ({sentinels}) THEN NULL ELSE past_due_30_59 END AS past_due_30_59,
    LEAST(debt_ratio, {debt_ratio_cap}) AS debt_ratio,
    monthly_income,
    open_credit_lines,
    CASE WHEN past_due_90 IN ({sentinels}) THEN NULL ELSE past_due_90 END AS past_due_90,
    real_estate_loans,
    CASE WHEN past_due_60_89 IN ({sentinels}) THEN NULL ELSE past_due_60_89 END AS past_due_60_89,
    dependents,
    (past_due_30_59 IN ({sentinels})
        OR past_due_90 IN ({sentinels})
        OR past_due_60_89 IN ({sentinels})) AS has_delinquency_code,
    (revolving_utilization > {util_cap}
        OR debt_ratio > {debt_ratio_cap}) AS util_was_capped
FROM {schema}.{table}

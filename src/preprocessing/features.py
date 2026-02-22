def select_features(df):
    """
    Keep only required features for modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with selected features
    """

    selected_columns = [
        "Revenue_per_Day",
        "Expense_per_Day",
        "Capital",
        "Debt_to_Revenue_Ratio",
        "Credit_Score",
        "Receivable_Days",
        "Payable_Days",
        "Inventory_Days",
        "Cash_Conversion_Cycle",
        "Cash_Inflow_Adjusted",
        "Cash_Outflow_Adjusted",
        "OD_Limit",
        "OD_Utilization",
        "Business_Type",
        "Category"
    ]

    df_cleaned = df[selected_columns].copy()

    return df_cleaned

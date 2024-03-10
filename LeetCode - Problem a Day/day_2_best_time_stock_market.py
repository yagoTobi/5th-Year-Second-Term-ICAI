def best_time_stock_market(prices):
    l_pointer, r_pointer = 0, 1
    max_profit = 0
    while r_pointer < len(prices):
        if prices[l_pointer] < prices[r_pointer]:
            profit = prices[r_pointer] - prices[l_pointer]
            max_profit = max(max_profit, profit)
        else:
            l_pointer = r_pointer
        r_pointer += 1
    return max_profit



tickers_list = ['MSFT', 'AAPL', 'TSLA', 'NVDA', 'PG', 'JPM', 'V', 'JNJ', 'FB']

#timeframe_list = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

frequency_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']



# çöp fonksşyon düzelt bunu
def get_tickers():


    # get symbols
    symbols = mt5.symbols_get()
    symbols_df = pd.DataFrame(symbols, columns=symbols[0]._asdict().keys())

    symbol_names = symbols_df['name'].tolist()
    return symbol_names
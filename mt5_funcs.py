import MetaTrader5 as mt5
import pandas as pd


TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
TIMEFRAME_DICT = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}


def get_symbol_names():
    # connect to MetaTrader5 platform
    # mt5.initialize() demo hesabına bağlanma için bilgiler
    # hesaba girmeyi denemeyin, içinde gerçek para yok :)
    if not mt5.initialize(login=58761515, server="MetaQuotes-Demo", password="koovt8zx"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # get symbols
    symbols = mt5.symbols_get()
    symbols_df = pd.DataFrame(symbols, columns=symbols[0]._asdict().keys())

    symbol_names = symbols_df['name'].tolist()
    return symbol_names


deneme_hesabi = (58761515, "koovt8zx", "MetaQuotes-Demo")
ziraat_deneme_hesabi = (99212315, "v7zxdtos", "ZiraatFX-Real")

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)
print(' ')

mt5.initialize()
symbols = mt5.symbols_get()
print(type(symbols))
print(len(symbols))
#print(symbols)
symbols_total=mt5.symbols_total()
print(symbols_total)
# for i in range(0,100):
#     print(symbols[i].path)


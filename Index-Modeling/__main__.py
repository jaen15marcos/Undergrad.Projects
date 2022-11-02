import datetime as dt

from index_model.index import IndexModel

if __name__ == "__main__":
    backtest_start = dt.date(year=2020, month=3, day=2)
    backtest_end = dt.date(year=2020, month=12, day=31)
    index = IndexModel()
    index.calc_index_level(start_date=backtest_start, end_date=backtest_end)
    index.export_values("export.csv")

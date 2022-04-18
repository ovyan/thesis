from __future__ import print_function, division
import sys
import pandas as pd
from nilmtk import DataSet
from nilmtk import MeterGroup
from nilmtk.elecmeter import ElecMeter


class BatchGenerator():
    TOKENIZED_SEQUENCE_FILENAME_TEMPLATE = "UKDALE-{}-{}-{}mins-building{}_tokenized.csv"

    def __init__(self, start_date, end_date, window, building, data_file):
        self.dataset = DataSet(data_file)
        self.__building = building
        self.__window = window
        self.__start_date = start_date
        self.__end_date = end_date
        self.__mins = window * 6 / 60
        self.__filename = self.get_filename()

    def get_filename(self):
        return self.TOKENIZED_SEQUENCE_FILENAME_TEMPLATE.format(self.__start_date, self.__end_date,
                                                                self.__mins, self.__building)

    def execute(self):
        series = self.__get_series()
        tokens_sequence = self.__tokenize_series(series)
        self.__save_to_csv(tokens_sequence)

    def __tokenize_series(self, series):
        series.fillna(0, inplace=True)
        print(series)
        print(type(series))
        print(series.values.shape)
        drop_values = len(series) % self.__window
        series.drop(series.tail(drop_values).index, inplace=True)
        tokens_sequence = series.values.reshape(-1, self.__window)
        print(tokens_sequence)
        print(tokens_sequence.shape)
        return tokens_sequence

    def __get_series(self):
        self.dataset.set_window(start=self.__start_date, end=self.__end_date)
        meter_group = self.dataset.buildings[self.__building].elec
        print("\n Meter Group: {}".format(meter_group))
        df_all_meters = meter_group.dataframe_of_meters()
        main_meters = meter_group.mains()
        return df_all_meters[main_meters.identifier][:]

    def __save_to_csv(self, tokens_sequence):
        df_tokens_sequence = pd.DataFrame(tokens_sequence)
        df_tokens_sequence.to_csv(self.__filename)

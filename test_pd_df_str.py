#!/usr/bin/env python3

import dataclasses
import inspect
import logging
import math
import random
import string
import typing

import pandas

import pd_df_str

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def timer(func: typing.Callable = None) -> typing.Callable:
    '''Timer decorator. Logs execution time for functions.''' # [Primer on Python Decorators](https://realpython.com/primer-on-python-decorators/)
    import functools
    import timeit
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = timeit.default_timer()
        value = func(*args, **kwargs)
        t1 = timeit.default_timer()
        logging.info(f'[{t1-t0:.6f} sec] {func.__module__}.{func.__name__}()')
        return value
    return wrapper


@dataclasses.dataclass
class Benchmark:
    # [Faster string processing in Pandas](https://www.gresearch.co.uk/article/faster-string-processing-in-pandas/)

    df: pandas.DataFrame = None
    args: typing.Dict = dataclasses.field(default_factory=list)
    kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    method: str = 'center'

    @staticmethod
    @timer
    def testDF(shape: typing.Tuple[int] = ((500,100,10), (2,2,2)), nCharVal: int = 10, nCharIdx: int = 20) -> pandas.DataFrame:
        return Generate.multiIndexDF(rowShape=shape[0], colShape=shape[1], nCharVal=nCharVal, nCharIdx=nCharIdx)

    @timer
    def apply(self, *args, **kwargs) -> pandas.DataFrame:
        return self.df.STR._apply(*args, **kwargs)

    @timer
    def applymap(self, *args, **kwargs) -> pandas.DataFrame:
        return self.df.STR._applymap(*args, **kwargs)

    @timer
    def npChar(self, *args, **kwargs) -> pandas.DataFrame:
        return self.df.STR._npChar(*args, **kwargs)

    @timer
    def pdConcat(self, *args, **kwargs) -> pandas.DataFrame:
        return self.df.STR._pdConcat(*args, **kwargs)

    @timer
    def dictComprehension(self, *args, **kwargs) -> pandas.DataFrame:
        return self.df.STR._dictComprehension(*args, **kwargs)

    @timer
    def stackUnstack(self, *args, **kwargs) -> pandas.DataFrame:
        return self.df.STR._stackUnstack(*args, **kwargs)

    def benchmark(self):
        self.df = self.testDF()
        self.df.STR._method = self.method
        self.args = list(Test.kwargs().get(self.method).values())
        self.kwargs = Test.kwargs().get(self.method)
        self.apply(**self.kwargs)
        self.dictComprehension(**self.kwargs)
        self.applymap(*self.args)
        self.npChar(**self.kwargs)
        self.pdConcat(**self.kwargs)
        self.stackUnstack(**self.kwargs)


@dataclasses.dataclass
class Generate:

    @staticmethod
    def airports(mixedType: bool = False) -> pandas.DataFrame:
        df = pandas.read_csv('https://raw.githubusercontent.com/vega/vega-datasets/next/data/airports.csv')
        return df if mixedType else df[['iata','name','city','state','country']]

    @staticmethod
    def randomString(nChar: int) -> str:
        return str.join('', random.choices(population=string.ascii_letters, k=nChar)) # [How to generate random strings in Python?](https://stackoverflow.com/a/2030293)

    @classmethod
    def df(cls, shape: typing.Tuple[int], nChar: int = 10, **kwargs) -> pandas.DataFrame:
        logging.info(f'generating random `pandas.DataFrame` with {shape[0]} rows and {shape[1]} columns...')
        data = ((cls.randomString(nChar=nChar) for i in range(shape[1])) for j in range(shape[0]))
        columns = list(string.ascii_letters)[:shape[1]] if (shape[1] <= len(string.ascii_letters)) else list(cls.randomString(nChar=shape[1]))
        return pandas.DataFrame(data=data, columns=columns, **kwargs)

    @classmethod
    def multiIndex(cls, shape: typing.Tuple[int], nChar: int = 4) -> pandas.MultiIndex:
        iterables = ((cls.randomString(nChar=nChar) for i in range(len)) for len in shape)
        return pandas.MultiIndex.from_product(iterables)

    @classmethod
    def multiIndexDF(cls, rowShape: typing.Tuple[int], colShape: typing.Tuple[int], nCharVal: int = 10, nCharIdx: int = 4) -> pandas.DataFrame:
        logging.info(f'generating random MultiIndex `pandas.DataFrame` with {math.prod(rowShape)} rows and {math.prod(colShape)} columns...')
        rowIndex = cls.multiIndex(shape=rowShape, nChar=nCharIdx) if isinstance(rowShape, tuple) else range(rowShape)
        colIndex = cls.multiIndex(shape=colShape, nChar=nCharIdx) if isinstance(colShape, tuple) else range(colShape)
        data = ((cls.randomString(nChar=nCharVal) for i,_ in enumerate(colIndex)) for j,_ in enumerate(rowIndex))
        return pandas.DataFrame(data=data, index=rowIndex, columns=colIndex)


@dataclasses.dataclass
class PandasType:
    # [ENH: Add dtype-support for pandas' type-hinting](https://github.com/pandas-dev/pandas/issues/34248)
    # [pandas.arrays.BooleanArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.BooleanArray.html)
    # [pandas.arrays.IntegerArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.IntegerArray.html)
    # [pandas.arrays.StringArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.StringArray.html)

    pandasObj = typing.Union[pandas.Index, pandas.Series, pandas.DataFrame]

    def pandasTypeVar(dtype: str):
        # [How to specify the type of pandas series elements in type hints?](https://stackoverflow.com/a/67493775)
        return typing.Union[typing.TypeVar(f'pandas.Index({dtype})'), typing.TypeVar(f'pandas.Series({dtype})'), typing.TypeVar(f'pandas.DataFrame({dtype})')]

    pandasBool = pandasTypeVar('bool')
    pandasInt = pandasTypeVar('int')
    pandasStr = pandasTypeVar('str')

    # [numpy.typing.NDArray](https://numpy.org/devdocs/reference/typing.html#numpy.typing.NDArray)
    import numpy.typing
    numpyBool = numpy.typing.NDArray[bool]
    numpyInt = numpy.typing.NDArray[int]
    numpyStr = numpy.typing.NDArray[str]


@dataclasses.dataclass
class Test:

    case: bool = False
    encoding: str = 'ascii'
    end: int = None
    errors: str = 'strict'
    expand: bool = True
    fillchar: str = '-'
    flags: int = 0
    form: str = 'NFC'
    i: int = 4
    join: str = 'left'
    n: int = -1
    na: str = None
    pat: str = 'a'
    regex: bool = True
    repeats: int = 4
    repl: str = 'z'
    sep: str = '|'
    side: str = 'left'
    start: int = 0
    step: int = None
    stop: int = None
    testDF: pandas.DataFrame = None
    testSeries: pandas.DataFrame = None
    to_strip: str = None # If `None` then whitespaces are removed
    width: int = 12
    wrap_expand_tabs: bool = True
    wrap_replace_whitespace: bool = True
    wrap_drop_whitespace: bool = True
    wrap_break_long_words: bool = True
    wrap_break_on_hyphens: bool = True
    KWARGS: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.testDF: pandas.DataFrame = Generate.multiIndexDF(rowShape=(2, 2, 2), colShape=(3, 2, 2, 2), nCharVal=10, nCharIdx=4)
        self.testSeries: pandas.DataFrame = Generate.df(shape=(40, 2))

    @classmethod
    def pandasKWARGS(cls, method: str):
        methodKwargs = inspect.signature(getattr(pandas.Series.str, method)).parameters
        return {k: v.default if v.default != inspect._empty else None for k,v in methodKwargs.items() if k != 'self'}

    @classmethod
    def kwargs(cls, **kwargs) -> typing.Dict:
        '''Get default kwargs for each method in `pandas.Series.str` and update them with `KWARGS` (in case more methods or kwargs are than are in `KWARGS` are added to `pandas.Series.str` in the future)'''
        # [String handling](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling)
        cls.KWARGS = {'capitalize': {},
                      'casefold': {},
                      'cat': {'others': kwargs.get('others'), 'sep': cls.sep, 'na_rep': cls.na, 'join': cls.join},
                      'center': {'width': cls.width, 'fillchar': cls.fillchar},
                      'contains': {'pat': cls.pat, 'case': cls.case, 'flags': cls.flags, 'na': cls.na, 'regex': cls.regex},
                      'count': {'pat': cls.pat, 'flags': cls.flags},
                      'decode': {'encoding': cls.encoding, 'errors': cls.errors},
                      'encode': {'encoding': cls.encoding, 'errors': cls.errors},
                      'endswith': {'pat': cls.pat, 'na': cls.na},
                      'extract': {'pat': f'({cls.pat})', 'flags': cls.flags, 'expand': cls.expand},
                      'extractall': {'pat': f'({cls.pat})', 'flags': cls.flags},
                      'find': {'sub': cls.pat, 'start': cls.start, 'end': cls.end},
                      'findall': {'pat': cls.pat, 'flags': cls.flags},
                      'fullmatch': {'pat': cls.pat, 'case': cls.case, 'flags': cls.flags, 'na': cls.na},
                      'get': {'i': cls.i},
                      'index': {'sub': cls.pat, 'start': cls.start, 'end': cls.end},
                      'join': {'sep': cls.sep},
                      'len': {},
                      'ljust': {'width': cls.width, 'fillchar': cls.fillchar},
                      'lower': {},
                      'lstrip': {'to_strip': cls.to_strip},
                      'match': {'pat': cls.pat, 'case': cls.case, 'flags': cls.flags, 'na': cls.na},
                      'normalize': {'form': cls.form},
                      'pad': {'width': cls.width, 'side': cls.side, 'fillchar': cls.fillchar},
                      'partition': {'sep': cls.pat, 'expand': cls.expand},
                      'removeprefix': {'prefix': cls.pat},
                      'removesuffix': {'suffix': cls.pat},
                      'repeat': {'repeats': cls.repeats},
                      'replace': {'pat': cls.pat, 'repl': cls.repl, 'n': cls.n, 'case': cls.case, 'flags': cls.flags, 'regex': cls.regex},
                      'rfind': {'sub': cls.pat, 'start': cls.start, 'end': cls.end},
                      'rindex': {'sub': cls.pat, 'start': cls.start, 'end': cls.end},
                      'rjust': {'width': cls.width, 'fillchar': cls.fillchar},
                      'rpartition': {'sep': cls.pat, 'expand': cls.expand},
                      'rstrip': {'to_strip': cls.to_strip},
                      'slice': {'start': cls.start, 'stop': cls.stop, 'step': cls.step},
                      'slice_replace': {'start': cls.start, 'stop': cls.stop, 'repl': cls.repl},
                      'split': {'pat': cls.pat, 'n': cls.n, 'expand': cls.expand, 'regex': cls.regex},
                      'rsplit': {'pat': cls.pat, 'n': cls.n, 'expand': cls.expand},
                      'startswith': {'pat': cls.pat, 'na': cls.na},
                      'strip': {'to_strip': cls.to_strip},
                      'swapcase': {},
                      'title': {},
                      'translate': {'table': str.maketrans('a', 'z')},
                      'upper': {},
                      'wrap': {'width': cls.width, 'expand_tabs': cls.wrap_expand_tabs, 'replace_whitespace': cls.wrap_replace_whitespace, 'drop_whitespace': cls.wrap_drop_whitespace, 'break_long_words': cls.wrap_break_long_words, 'break_on_hyphens': cls.wrap_break_on_hyphens},
                      'zfill': {'width': cls.width},
                      'isalnum': {},
                      'isalpha': {},
                      'isdigit': {},
                      'isspace': {},
                      'islower': {},
                      'isupper': {},
                      'istitle': {},
                      'isnumeric': {},
                      'isdecimal': {},
                      'get_dummies': {'sep': cls.sep}
                      }
        pandasKWARGS = {method: cls.pandasKWARGS(method=method) for method in dir(pandas.Series.str) if not method.startswith('_')}
        return {**pandasKWARGS, **cls.KWARGS}

    def dataFrame(self):
        df = self.testDF
        kwargs = self.kwargs(others=df.iloc[:, 0])
        for method in dir(df.STR):
            if not method.startswith('_') and method not in ('index', 'rindex'):
                logging.info(method)
                df.STR._method = method
                assert isinstance(getattr(df.STR, method)(**kwargs.get(method)), pandas.DataFrame)

    def series(self):
        df = self.testSeries
        kwargs = self.kwargs(others=df.iloc[:, 0])
        s = df.iloc[:, 0]
        for method in dir(df.STR):
            if not method.startswith('_') and method not in ('index', 'rindex'):
                logging.info(method)
                s.STR._method = method
                reference = getattr(s.str, method)(**kwargs.get(method))
                testMethod = pandas.testing.assert_frame_equal if isinstance(reference, pandas.DataFrame) else pandas.testing.assert_series_equal if isinstance(reference, (pandas.Index, pandas.Series)) else None
                testMethod(reference, getattr(s.STR, method)(**kwargs.get(method)))

    def indexMethod(self):
        df = self.testDF
        logging.info('index')
        assert (df.STR.join(sep=self.sep).STR.index(sub=self.sep) == 1).all().all()
        if df.STR.len().all().all():
            logging.info('rindex')
            strLen = 2 * len(df.iloc[0].iloc[0]) - 2 - 1
            assert (df.STR.join(sep=self.sep).STR.rindex(sub=self.sep) == strLen).all().all()


def main():
    test = Test()
    test.dataFrame()
    test.series()
    test.indexMethod()

if __name__ == '__main__':
    main()

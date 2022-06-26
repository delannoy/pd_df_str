#!/usr/bin/env python3

import dataclasses
import inspect
import math
import random
import string
import typing

import pandas

import pd_df_str


@dataclasses.dataclass
class Benchmark:
    # [Faster string processing in Pandas](https://www.gresearch.co.uk/article/faster-string-processing-in-pandas/)

    @staticmethod
    def testDF(shape: typing.Tuple[int] = ((500,100,10), (2,2,2)), nCharVal: int = 10, nCharIdx: int = 20) -> pandas.DataFrame:
        print(f'generating random `pandas.DataFrame` with {math.prod(shape[0])} rows and {math.prod(shape[1])} columns...')
        return Generate.multiIndexDF(rowShape=shape[0], colShape=shape[1], nCharVal=nCharVal, nCharIdx=nCharIdx)

    @staticmethod
    def applyLambda(df: pandas.DataFrame, method: str, *args, **kwargs) -> pandas.DataFrame:
        print(f'df.apply(lambda col: col.str.{method}({args}, {kwargs}))')
        return df.STR._applyLambda(df, method, *args, **kwargs)

    @staticmethod
    def applyMap(df: pandas.DataFrame, method: str, *args, **kwargs) -> pandas.DataFrame:
        print(f'df.applymap(lambda element: str.{method}(element, {args}, {kwargs}))')
        return df.STR._applyMap(df, method, *args, **kwargs)

    @staticmethod
    def applyNumpyChar(df: pandas.DataFrame, method: str, *args, **kwargs) -> pandas.DataFrame:
        print(f'df.apply(lambda col: numpy.char.{method}(col.to_numpy(), {args}, {kwargs}))')
        return df.STR._applyNumpyChar(df, method, *args, **kwargs)

    @staticmethod
    def concat(df: pandas.DataFrame, method: str, *args, **kwargs) -> pandas.DataFrame:
        print(f'pandas.concat([df[col].str.{method}({args}, {kwargs}) for col in df])')
        return df.STR._concat(df, method, *args, **kwargs)

    @staticmethod
    def dictComprehension(df: pandas.DataFrame, method: str, *args, **kwargs) -> pandas.DataFrame:
        print(f'pandas.DataFrame({{col: df[col].str.{method}({args}, {kwargs}) for col in df}})')
        return df.STR._dictComprehension(df, method, *args, **kwargs)

    @staticmethod
    def stackUnstack(df: pandas.DataFrame, method: str, *args, **kwargs) -> pandas.DataFrame:
        print(f'col.stack.().str.{method}({args}, {kwargs}).unstack()')
        return df.STR._stackUnstack(df, method, *args, **kwargs)

# df = Benchmark.testDF()
# %timeit Benchmark.applyLambda(df, 'center', 20, '-')
# %timeit Benchmark.dictComprehension(df, 'center', 20, '-')
# %timeit Benchmark.applyMap(df, 'center', 20, '-')
# %timeit Benchmark.applyNumpyChar(df, 'center', 20, '-')
# %timeit Benchmark.concat(df, 'center', 20, '-')
# %timeit Benchmark.stackUnstack(df, 'center', 20, '-')

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
    def df(cls, shape: typing.Tuple[int], nChar: int = 8, **kwargs) -> pandas.DataFrame:
        return pandas.DataFrame(data=[[cls.randomString(nChar=nChar) for i in range(shape[1])] for j in range(shape[0])], **kwargs)

    @classmethod
    def multiIndex(cls, shape: typing.Tuple[int], nChar: int = 4) -> pandas.MultiIndex:
        iterables = [[cls.randomString(nChar=nChar) for i in range(len)] for len in shape]
        return pandas.MultiIndex.from_product(iterables)

    @classmethod
    def multiIndexDF(cls, rowShape: typing.Tuple[int], colShape: typing.Tuple[int], nCharVal: int = 10, nCharIdx: int = 4) -> pandas.DataFrame:
        rowIndex = cls.multiIndex(shape=rowShape, nChar=nCharIdx) if isinstance(rowShape, tuple) else range(rowShape)
        colIndex = cls.multiIndex(shape=colShape, nChar=nCharIdx) if isinstance(colShape, tuple) else range(colShape)
        data = [[cls.randomString(nChar=nCharVal) for i,_ in enumerate(colIndex)] for j,_ in enumerate(rowIndex)]
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
    to_strip: str = None # If None then whitespaces are removed
    width: int = 12
    wrap_expand_tabs: bool = True
    wrap_replace_whitespace: bool = True
    wrap_drop_whitespace: bool = True
    wrap_break_long_words: bool = True
    wrap_break_on_hyphens: bool = True
    KWARGS: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def pdKwargs(cls, method: str):
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
        pdKwargs = {method: cls.pdKwargs(method=method) for method in dir(pandas.Series.str) if not method.startswith('_')}
        return {**pdKwargs, **cls.KWARGS}

    @classmethod
    def dataFrame(cls, df: pandas.DataFrame = Generate.multiIndexDF(rowShape=(2, 2, 2), colShape=(3, 2, 2, 2))):
        kwargs = cls.kwargs(others=df.iloc[:, 0])
        for method in dir(pd_df_str.StringAccessor):
            if not method.startswith('_') and method not in ('index', 'rindex'):
                print(method)
                assert isinstance(getattr(df.STR, method)(**kwargs.get(method)), pandas.DataFrame)

    @classmethod
    def series(cls, df: pandas.DataFrame = Generate.df(shape=(40, 2), columns=['a','b'])):
        kwargs = cls.kwargs(others=df.iloc[:, 0])
        for method in dir(pd_df_str.StringAccessor):
            if not method.startswith('_') and method not in ('index', 'rindex'):
                print(method)
                reference = getattr(df.iloc[:, 0].str, method)(**kwargs.get(method))
                testMethod = pandas.testing.assert_frame_equal if isinstance(reference, pandas.DataFrame) else pandas.testing.assert_series_equal if isinstance(reference, (pandas.Index, pandas.Series)) else None
                testMethod(reference, getattr(df.iloc[:, 0].STR, method)(**kwargs.get(method)))

    @staticmethod
    def indexMethod(sep: str = '-', df: pandas.DataFrame = Generate.multiIndexDF(rowShape=(2, 2, 2), colShape=(3, 2, 2, 2), nCharVal=10)):
        print('index')
        assert (df.STR.join(sep=sep).STR.index(sub=sep) == 1).all().all()
        if df.STR.len().all().all():
            print('rindex')
            strLen = 2 * len(df.iloc[0].iloc[0]) - 2 - 1
            assert (df.STR.join(sep=sep).STR.rindex(sub=sep) == strLen).all().all()


def main():
    Test.dataFrame()
    Test.series()
    Test.indexMethod()

if __name__ == '__main__':
    main()

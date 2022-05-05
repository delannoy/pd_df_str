#!/usr/bin/env python3

import dataclasses
import inspect
import typing

import numpy
import pandas

accessorName = 'STR' # [Registering custom accessors](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors)


@pandas.api.extensions.register_dataframe_accessor(accessorName)
@pandas.api.extensions.register_series_accessor(accessorName)
@pandas.api.extensions.register_index_accessor(accessorName)
@dataclasses.dataclass
class StringAccessor:

    '''
    This class generalizes `pandas.Series.str` methods so that they can be applied to `pandas.DataFrame` objects.
    For simple methods (i.e. methods that return a `pandas.Series` when given a `pandas.Series` as input), a `pandas.DataFrame` is produced from a dictionary comprehension that calls the method across all columns.
    For complex methods (i.e. methods that return a `pandas.DataFrame` when given a `pandas.Series` as input), the method is called across all columns and the resulting objects are concatenated together into an output `pandas.DataFrame`.
    '''

    # [String handling](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling)
    # [Working with text data](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)

    _obj: typing.Union[pandas.Index, pandas.Series, pandas.DataFrame]

    def __post_init__(self, _validate: bool = True):
        '''Delegate object validation to `pandas.Series.str._validate` function for each column in the input `pandas.DataFrame`'''
        if _validate and isinstance(self._obj, pandas.DataFrame):
            [pandas.Series.str._validate(self._obj[col]) for col in self._obj.columns]

    def __getitem__(self, idx: int):
        '''[Called to implement evaluation of `self[key]`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__)'''
        # [Indexing with .str](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#indexing-with-str)
        return self.get(idx)

    def _apply(self, _concat: bool = False, **kwargs) -> typing.Union[pandas.Index, pandas.Series, pandas.DataFrame]:
        '''Execute the `pandas.Series.str._method` (where `_method` corresponds to the name of the parent function) on an arbitratry `pandas` object'''
        _method = inspect.stack()[1].function # [Getting the caller function name inside another function in Python?](https://stackoverflow.com/a/900404/13019084)
        if isinstance(self._obj, (pandas.Index, pandas.Series)):
            return getattr(self._obj.str, _method)(**kwargs)
        elif isinstance(self._obj, pandas.DataFrame):
            if _concat:
                return self._concat(obj=self._obj, _method=_method, **kwargs)
            else:
                return self._dictComprehension(obj=self._obj, _method=_method, **kwargs)
                # return self._applyLambda(obj=self._obj, _method=_method, **kwargs)
                # return self._stackUnstack(obj=self._obj, _method=_method, **kwargs)

    @staticmethod
    def _applyLambda(obj: pandas.DataFrame, _method: str, *args, **kwargs) -> pandas.DataFrame:
        '''Apply `_method` along every column.'''
        return obj.apply(lambda col: getattr(col.str, _method)(*args, **kwargs), axis='index')

    @staticmethod
    def _applyMap(obj: pandas.DataFrame, _method: str, *args, **kwargs) -> pandas.DataFrame:
        '''Apply `_method` elementwise. Note that not all `pandas.Series.str` methods are available in, or consistent with, the python standard library.'''
        # [Built-in string methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
        obj = obj.astype(str)
        return obj.applymap(lambda element: getattr(element, _method)(*args, **kwargs), na_action='ignore')

    @staticmethod
    def _applyNumpyChar(obj: pandas.DataFrame, _method: str, *args, **kwargs) -> pandas.DataFrame:
        '''Apply `numpy.char._method` along each column. Note that not all `pandas.Series.str` methods are available in , or consistent with, `numpy.char` string operations.'''
        # [The numpy.char module provides a set of vectorized string operations for arrays of type numpy.str_ or numpy.bytes_. All of them are based on the string methods in the Python standard library.](https://numpy.org/doc/stable/reference/routines.char.html)
        # return obj.apply(lambda col: getattr(numpy.char, _method)(col.array.astype(str), *args, **kwargs))
        # return obj.apply(lambda col: getattr(numpy.char, _method)(numpy.array(col, dtype=str), *args, **kwargs))
        return obj.apply(lambda col: getattr(numpy.char, _method)(col.to_numpy(dtype=str), *args, **kwargs))

    @staticmethod
    def _concat(obj: pandas.DataFrame, _method: str, *args, _delim: str = '|', **kwargs) -> pandas.DataFrame:
        '''`_method` is called for every column and the resulting objects are concatenated together into an output `pandas.DataFrame`'''
        df = pandas.DataFrame()
        for col in obj.columns:
            _obj = getattr(obj[col].str, _method)(*args, **kwargs)
            col = str.join(_delim, col) if isinstance(col, tuple) else col # merge MultiIndex column tuple into a string using `_delim` as a delimiter
            _obj = _obj.add_prefix(f'{col}_') if isinstance(_obj, pandas.DataFrame) else _obj.rename(col) # prefix the column name if applying `_method` to `obj[col]` results in a `pandas.DataFrame`, else set as label/name for the `pandas.Series`
            df = pandas.concat([df, _obj], axis='columns')
        if isinstance(obj.columns, pandas.MultiIndex):
            df.columns = pandas.MultiIndex.from_tuples(col.split(_delim) for col in df.columns) # [Create Multiindex from pattern in column names](https://stackoverflow.com/a/37242458/13019084)
        return df

    @staticmethod
    def _dictComprehension(obj: pandas.DataFrame, _method: str, *args, **kwargs) -> pandas.DataFrame:
        '''`_method` is called across every column within a dictonary comprehension and the result input into a `pandas.DataFrame`'''
        # [How to apply string methods to multiple columns of a dataframe](https://stackoverflow.com/a/52099411/13019084)
        return pandas.DataFrame({col: (getattr(obj[col].str, _method)(*args, **kwargs)) for col in obj})

    @staticmethod
    def _stackUnstack(obj: pandas.DataFrame, _method: str, *args, **kwargs) -> pandas.DataFrame:
        '''`obj` is stacked into a `pandas.Series` (columns are "folded" or "pivoted" into the index), `_method` is applied to the resulting `pandas.Series`, and it is finally unstacked into the same shape as the input `pandas.DataFrame`'''
        # [Reshaping by stacking and unstacking](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-stacking-and-unstacking)
        idxLevel = [*range(obj.index.nlevels)] if obj.index.nlevels > 1 else [0]
        colLevel = [*range(obj.columns.nlevels)] if obj.columns.nlevels > 1 else [0]
        unstackLevel = [i+len(idxLevel) for i in colLevel] # only MultiIndex columns (not MultiIndex rows) need to be unstacked
        _obj = getattr(obj.stack(level=colLevel, dropna=False).str, _method)(*args, **kwargs)
        if isinstance(_obj, (pandas.Index, pandas.Series, pandas.DataFrame)): # `pandas.Series.str.cat` can return a goddamn `str` instead of a pandas object
            return _obj.unstack(level=unstackLevel).reindex(index=obj.index, columns=obj.columns) # [How to maintain Pandas DataFrame index order when using stack/unstack?](https://stackoverflow.com/a/33608397/13019084)
        return _obj

    def capitalize(self) -> pandas.arrays.StringArray:
        return self._apply()

    def casefold(self) -> pandas.arrays.StringArray:
        return self._apply()

    def cat(self, others: typing.Union[typing.List, typing.Set, typing.Tuple, pandas.Index, pandas.Series, pandas.DataFrame] = None, sep: str = None, na_rep: typing.Union[str, None] = None, join: str = 'left') -> typing.Union[str, pandas.arrays.StringArray]:
        assert join in ('left', 'right', 'outer', 'inner')
        _concat = False if others is None else True
        return self._apply(others=others, sep=sep, na_rep=na_rep, join=join, _concat=_concat)

    def center(self, width: int, fillchar: str = ' ') -> pandas.arrays.StringArray:
        return self._apply(width=width, fillchar=fillchar)

    def contains(self, pat: str, case: bool = True, flags: int = 0, na: typing.Union[str, float, pandas._libs.missing.NAType] = None, regex: bool = True) -> pandas.arrays.BooleanArray:
        # flags: typing.Union[int, re.ASCII, re.DEBUG, re.IGNORECASE, re.LOCALE, re.MULTILINE, re.DOTALL, re.VERBOSE]
        return self._apply(pat=pat, case=case, flags=flags, na=na, regex=regex)

    def count(self, pat: str, flags: int = 0, **kwargs) -> pandas.arrays.IntegerArray:
        return self._apply(pat=pat, flags=flags, **kwargs)

    def decode(self, encoding: str, errors: str = 'strict', _assert: bool = False) -> pandas.arrays.StringArray:
        assert errors in ('strict', 'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplace', 'namereplace', 'surrogateescape') # [Error Handlers](https://docs.python.org/3/library/codecs.html#error-handlers)
        if _assert:
            assert endoding in set(pandas.read_html('https://docs.python.org/3/library/codecs.html#standard-encodings', flavor='lxml', match='Aliases')[0]['Aliases'].str.casefold().str.split(', ', expand=True).stack())
        return self._apply(encoding=encoding, errors=errors)

    def encode(self, encoding: str, errors: str = 'strict', _assert: bool = False) -> pandas.arrays.StringArray:
        assert errors in ('strict', 'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplace', 'namereplace', 'surrogateescape') # [Error Handlers](https://docs.python.org/3/library/codecs.html#error-handlers)
        if _assert:
            assert endoding in set(pandas.read_html('https://docs.python.org/3/library/codecs.html#standard-encodings', flavor='lxml', match='Aliases')[0]['Aliases'].str.casefold().str.split(', ', expand=True).stack())
        return self._apply(encoding=encoding, errors=errors)

    def endswith(self, pat: str, na: typing.Union[str, float, pandas._libs.missing.NAType] = None) -> pandas.arrays.BooleanArray:
        return self._apply(pat=pat, na=na)

    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> pandas.arrays.StringArray:
        # `pandas.Series.str.extract` can return a `Series` or `DataFrame` depending on the value the `expand` kwarg... _if_ there is only one capture group!!!
        return self._apply(pat=pat, flags=flags, expand=expand, _concat=True)

    def extractall(self, pat: str, flags: int = 0) -> pandas.DataFrame:
        return self._apply(pat=pat, flags=flags, _concat=True)

    def find(self, sub: str, start: int = 0, end: typing.Union[int, None] = None) -> pandas.arrays.IntegerArray:
        return self._apply(sub=sub, start=start, end=end)

    def findall(self, pat: str, flags: int = 0) -> pandas.arrays.StringArray:
        return self._apply(pat=pat, flags=flags)

    def fullmatch(self, pat: str, case: bool = True, flags: int = 0, na: typing.Union[str, float, pandas._libs.missing.NAType] = None) -> pandas.arrays.BooleanArray:
        return self._apply(pat=pat, case=case, flags=flags, na=na)

    def get(self, i: int) -> pandas.arrays.StringArray:
        return self._apply(i=i)

    def index(self, sub: str, start: int = 0, end: typing.Union[int, None] = None) -> pandas.arrays.IntegerArray:
        return self._apply(sub=sub, start=start, end=end)

    def join(self, sep: str) -> pandas.arrays.StringArray:
        return self._apply(sep=sep)

    def len(self) -> pandas.arrays.IntegerArray:
        return self._apply()

    def ljust(self, width: int, fillchar: str = ' ') -> pandas.arrays.StringArray:
        return self._apply(width=width, fillchar=fillchar)

    def lower(self) -> pandas.arrays.StringArray:
        return self._apply()

    def lstrip(self, to_strip: typing.Union[str, None] = None) -> pandas.arrays.StringArray:
        return self._apply(to_strip=to_strip)

    def match(self, pat: str, case: bool = True, flags: int = 0, na: typing.Union[str, float, pandas._libs.missing.NAType] = None) -> pandas.arrays.BooleanArray:
        return self._apply(pat=pat, case=case, flags=flags, na=na)

    def normalize(self, form: str) -> pandas.arrays.StringArray:
        assert form in ('NFC', 'NFKC', 'NFD', 'NFKD')
        return self._apply(form=form)

    def pad(self, width: int, side: int = 'left', fillchar: str = ' ') -> pandas.arrays.StringArray:
        assert side in ('left', 'right', 'both')
        return self._apply(width=width, side=side, fillchar=fillchar)

    def partition(self, sep: str = ' ', expand: bool = True) -> pandas.arrays.StringArray:
        return self._apply(sep=sep, expand=expand, _concat=expand)

    def removeprefix(self, prefix: str) -> pandas.arrays.StringArray:
        return self._apply(prefix=prefix)

    def removesuffix(self, suffix: str) -> pandas.arrays.StringArray:
        return self._apply(suffix=suffix)

    def repeat(self, repeats: typing.Union[int, typing.List[int]]) -> pandas.arrays.StringArray:
        return self._apply(repeats=repeats)

    def replace(self, pat: str, repl: str, n: int = -1, case: bool = None, flags: int = 0, regex:bool = True) -> pandas.arrays.StringArray:
        return self._apply(pat=pat, repl=repl, n=n, case=case, flags=flags, regex=regex)

    def rfind(self, sub: str, start: int = 0, end: typing.Union[int, None] = None) -> pandas.arrays.IntegerArray:
        return self._apply(sub=sub, start=start, end=end)

    def rindex(self, sub: str, start: int = 0, end: typing.Union[int, None] = None) -> pandas.arrays.IntegerArray:
        return self._apply(sub=sub, start=start, end=end)

    def rjust(self, width: int, fillchar: str = ' ') -> pandas.arrays.StringArray:
        return self._apply(width=width, fillchar=fillchar)

    def rpartition(self, sep: str = ' ', expand: bool = True) -> pandas.arrays.StringArray:
        return self._apply(sep=sep, expand=expand, _concat=expand)

    def rstrip(self, to_strip: typing.Union[str, None] = None) -> pandas.arrays.StringArray:
        return self._apply(to_strip=to_strip)

    def slice(self, start: int = None, stop: int = None, step: int = None) -> pandas.arrays.StringArray:
        return self._apply(start=start, stop=stop, step=step)

    def slice_replace(self, start: int = None, stop: int = None, repl: str = None)-> pandas.arrays.StringArray:
        return self._apply(start=start, stop=stop, repl=repl)

    def split(self, pat: str=None, n: int=-1, expand: bool=False, *, regex: bool=False) -> pandas.arrays.StringArray:
        return self._apply(pat=pat, n=n, expand=expand, regex=regex, _concat=expand)

    def rsplit(self, pat: str=None, n: int=-1, expand: bool=False) -> pandas.arrays.StringArray:
        return self._apply(pat=pat, n=n, expand=expand, _concat=expand)

    def startswith(self, pat: str, na: typing.Union[str, float, pandas._libs.missing.NAType] = None) -> pandas.arrays.BooleanArray:
        return self._apply(pat=pat, na=na)

    def strip(self, to_strip: typing.Union[str, None] = None) -> pandas.arrays.StringArray:
        return self._apply(to_strip=to_strip)

    def swapcase(self) -> pandas.arrays.StringArray:
        return self._apply()

    def title(self) -> pandas.arrays.StringArray:
        return self._apply()

    def translate(self, table: typing.Dict) -> pandas.arrays.StringArray:
        return self._apply(table=table)

    def upper(self) -> pandas.arrays.StringArray:
        return self._apply()

    def wrap(self, width: int, expand_tabs: bool = True, replace_whitespace: bool = True, drop_whitespace: bool = True, break_long_words: bool = True, break_on_hyphens: bool = True, **kwargs) -> pandas.arrays.StringArray:
        return self._apply(width=width, expand_tabs=expand_tabs, replace_whitespace=replace_whitespace, drop_whitespace=drop_whitespace, break_long_words=break_long_words, break_on_hyphens=break_on_hyphens, **kwargs)

    def zfill(self, width: int) -> pandas.arrays.StringArray:
        return self._apply(width=width)

    def isalnum(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def isalpha(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def isdigit(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def isspace(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def islower(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def isupper(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def istitle(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def isnumeric(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def isdecimal(self) -> pandas.arrays.BooleanArray:
        return self._apply()

    def get_dummies(self, sep: str = '|') -> pandas.arrays.IntegerArray:
        return self._apply(sep=sep, _concat=True)

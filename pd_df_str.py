#!/usr/bin/env python3

import dataclasses
import functools
import typing

import pandas

accessorName = 'STR' # [Registering custom accessors](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors)


@pandas.api.extensions.register_dataframe_accessor(accessorName)
@pandas.api.extensions.register_series_accessor(accessorName)
@pandas.api.extensions.register_index_accessor(accessorName)
@dataclasses.dataclass
class StringAccessor:

    '''
    This class generalizes `pandas.Series.str` methods so that they can be applied to `pandas.DataFrame` objects.
    For simple `pandas.Series.str` methods (i.e. methods that return a `pandas.Series` when given a `pandas.Series` as input), the method is called across all columns via `pandas.DataFrame.apply`.
    For complex `pandas.Series.str` methods (i.e. methods that return a `pandas.DataFrame` when given a `pandas.Series` as input), the method is called across all columns in an explicit loop and the resulting objects are concatenated together into an output `pandas.DataFrame`.
    '''

    # [String handling](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling)
    # [Working with text data](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)

    _obj: typing.Union[pandas.Index, pandas.Series, pandas.DataFrame]
    _delim: str = '|'
    _method: str = None

    def __post_init__(self, validate: bool = True):
        '''Validate `self._obj` using `pandas.Series.str._validate`'''
        if validate:
            if isinstance(self._obj, pandas.Series):
                pandas.Series.str._validate(self._obj)
            if isinstance(self._obj, pandas.DataFrame):
                [pandas.Series.str._validate(self._obj[col]) for col in self._obj.columns] # Delegate object validation to `pandas.Series.str._validate` function for each column in the input `pandas.DataFrame`

    def __dir__(self) -> typing.List[str]:
        '''Update `StringAccessor` with `pandas.Series.str` methods'''
        return [*dir(StringAccessor), *[method for method in dir(pandas.Series.str) if not method.startswith('_')]] # [Calling builtin dir from overrided dir method in Python](https://stackoverflow.com/a/50199953/13019084)

    def __getattr__(self, _method: str):
        '''Intercept method calls and store the method name as an attribute, `self._method`''' # [Python: How do you intercept a method call to change function parameters?](https://stackoverflow.com/a/44954278/13019084)
        func = getattr(pandas.Series.str, _method)
        self._method = _method

        @functools.wraps(func)
        def _intercept(*args, **kwargs) -> typing.Union[pandas.Index, pandas.Series, pandas.DataFrame]:
            '''Execute `pandas.Series.str.{_method}` on an arbitratry `pandas` object'''
            if isinstance(self._obj, (pandas.Index, pandas.Series)):
                return getattr(self._obj.str, _method)(*args, **kwargs)
            if isinstance(self._obj, pandas.DataFrame):
                if _method in ('extract', 'extractall', 'partition', 'rpartition', 'split', 'rsplit', 'get_dummies'):
                    return self._pdConcat(*args, **kwargs)
                else:
                    return self._apply(*args, **kwargs)

        return _intercept

    def __getitem__(self, key: typing.Union[int, slice]) -> typing.Union[pandas.Index, pandas.Series, pandas.DataFrame]:
        '''Allows use of `[]` notation to directly index by position locations''' # [Implementing slicing in __getitem__](https://gaopinghuang0.github.io/2018/11/17/python-slicing) # [Indexing with .str](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#indexing-with-str)
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        elif isinstance(key, int):
            return self.get(key)

    def _apply(self, *args, **kwargs) -> pandas.DataFrame:
        '''Applies `_method` along every column.'''
        return self._obj.apply(lambda col: getattr(col.str, self._method)(*args, **kwargs), axis='index')

    def _pdConcat(self, *args, **kwargs) -> pandas.DataFrame:
        '''Calls `_method` for every column and the resulting objects are concatenated together into an output `pandas.DataFrame`'''
        df = pandas.DataFrame()
        for col in self._obj.columns:
            obj = getattr(self._obj[col].str, self._method)(*args, **kwargs)
            _col = str.join(self._delim, col) if isinstance(col, tuple) else col # merge MultiIndex column tuple into a string using `_delim` as a delimiter
            # print(_col)
            obj = obj.add_prefix(f'{_col}_') if isinstance(obj, pandas.DataFrame) else obj.rename(_col) if isinstance(obj, pandas.Series) else obj # prefix the column name if applying `_method` to `obj[col]` results in a `pandas.DataFrame`
            df = pandas.concat([df, obj], axis=1)
        if isinstance(self._obj.columns, pandas.MultiIndex):
            # [print(col.split(self._delim)) for col in df.columns]
            df.columns = pandas.MultiIndex.from_tuples(col.split(self._delim) for col in df.columns) # [Create Multiindex from pattern in column names](https://stackoverflow.com/a/37242458/13019084)
        return df

    def _dictComprehension(self, *args, **kwargs) -> pandas.DataFrame:
        '''Calls `_method` across every column within a dictonary comprehension''' # [How to apply string methods to multiple columns of a dataframe](https://stackoverflow.com/a/52099411/13019084)
        return pandas.DataFrame({col: (getattr(self._obj[col].str, self._method)(*args, **kwargs)) for col in self._obj})

    def _applymap(self, *args, asStr: bool = False, **kwargs) -> pandas.DataFrame:
        '''Applies `method` elementwise. Note that not all `pandas.Series.str` methods are available in, or consistent with, the python standard library.''' # [Built-in string methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
        if asStr:
            self._obj = self._obj.astype(str)
        return self._obj.applymap(lambda element: getattr(element, self._method)(*args, **kwargs), na_action='ignore')

    def _npChar(self, *args, **kwargs) -> pandas.DataFrame:
        '''Applies `numpy.char.{_method}` along each column. Note that not all `pandas.Series.str` methods are available in , or consistent with, `numpy.char` string operations.''' # [The numpy.char module provides a set of vectorized string operations for arrays of type numpy.str_ or numpy.bytes_. All of them are based on the string methods in the Python standard library.](https://numpy.org/doc/stable/reference/routines.char.html)
        import numpy
        def strArray(col: pandas.Series) -> numpy.ndarray:
            return col.to_numpy(dtype=str) # col.array.astype(str) # numpy.array(col, dtype=str)
        return self._obj.apply(lambda col: getattr(numpy.char, self._method)(strArray(col), *args, **kwargs))

    def _stackUnstack(self, *args, **kwargs) -> pandas.DataFrame:
        '''Stacks `self._obj` into a `pandas.Series` (columns are "folded" or "pivoted" into the index), `_method` is applied to the resulting `pandas.Series`, and it is then unstacked into the same shape as the input `pandas.DataFrame`''' # [Reshaping by stacking and unstacking](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-stacking-and-unstacking)
        idxLevel = [*range(self._obj.index.nlevels)] if self._obj.index.nlevels > 1 else [0]
        colLevel = [*range(self._obj.columns.nlevels)] if self._obj.columns.nlevels > 1 else [0]
        unstackLevel = [i+len(idxLevel) for i in colLevel] # only MultiIndex columns (not MultiIndex rows) need to be unstacked
        _obj = getattr(self._obj.stack(level=colLevel, dropna=False).str, self._method)(*args, **kwargs)
        if isinstance(_obj, (pandas.Index, pandas.Series, pandas.DataFrame)): # `pandas.Series.str.cat` returns a goddamn `str` instead of a pandas object when `others` is `None`
            return _obj.unstack(level=unstackLevel).reindex(index=self._obj.index, columns=self._obj.columns) # [How to maintain Pandas DataFrame index order when using stack/unstack?](https://stackoverflow.com/a/33608397/13019084)
        return _obj

# pd_df_str

Extends `pandas.Series.str` methods so that they can be applied to `pandas.DataFrame` objects by registering a custom accessor.
For simple `pandas.Series.str` methods (i.e. methods that return a `pandas.Series` when given a `pandas.Series` as input), the method is called across all columns via `pandas.DataFrame.apply`.
For complex `pandas.Series.str` methods (i.e. methods that return a `pandas.DataFrame` when given a `pandas.Series` as input), the method is called across all columns and the resulting objects are concatenated together into an output `pandas.DataFrame`.
MultiIndexed `pandas.DataFrame` objects are supported.

## Usage:
```python
import pandas
import pd_df_str
df = pandas.read_csv('https://raw.githubusercontent.com/vega/vega-datasets/next/data/airports.csv').drop(columns=['latitude','longitude'])
```

```python
df.STR.upper()
```
```
     iata                       name              city state country
0     00M                    THIGPEN       BAY SPRINGS    MS     USA
1     00R       LIVINGSTON MUNICIPAL        LIVINGSTON    TX     USA
2     00V                MEADOW LAKE  COLORADO SPRINGS    CO     USA
3     01G               PERRY-WARSAW             PERRY    NY     USA
4     01J           HILLIARD AIRPARK          HILLIARD    FL     USA
...   ...                        ...               ...   ...     ...
3371  ZEF            ELKIN MUNICIPAL             ELKIN    NC     USA
3372  ZER  SCHUYLKILL CTY/JOE ZERBEY        POTTSVILLE    PA     USA
3373  ZPH      ZEPHYRHILLS MUNICIPAL       ZEPHYRHILLS    FL     USA
3374  ZUN                 BLACK ROCK              ZUNI    NM     USA
3375  ZZV       ZANESVILLE MUNICIPAL        ZANESVILLE    OH     USA

[3376 rows x 5 columns]
```

```python
df.STR[1:6]
```
```
     iata   name   city state country
0      0M  higpe  ay Sp     S      SA
1      0R  iving  iving     X      SA
2      0V  eadow  olora     O      SA
3      1G  erry-   erry     Y      SA
4      1J  illia  illia     L      SA
...   ...    ...    ...   ...     ...
3371   EF  lkin    lkin     C      SA
3372   ER  chuyl  ottsv     A      SA
3373   PH  ephyr  ephyr     L      SA
3374   UN  lack     uni     M      SA
3375   ZV  anesv  anesv     H      SA

[3376 rows x 5 columns]
```

```python
df.STR.cat(df.city, sep='-')
```
```
                      iata                                  name                               city                state               country
0          00M-Bay Springs                   Thigpen-Bay Springs            Bay Springs-Bay Springs       MS-Bay Springs       USA-Bay Springs
1           00R-Livingston       Livingston Municipal-Livingston              Livingston-Livingston        TX-Livingston        USA-Livingston
2     00V-Colorado Springs          Meadow Lake-Colorado Springs  Colorado Springs-Colorado Springs  CO-Colorado Springs  USA-Colorado Springs
3                01G-Perry                    Perry-Warsaw-Perry                        Perry-Perry             NY-Perry             USA-Perry
4             01J-Hilliard             Hilliard Airpark-Hilliard                  Hilliard-Hilliard          FL-Hilliard          USA-Hilliard
...                    ...                                   ...                                ...                  ...                   ...
3371             ZEF-Elkin                 Elkin Municipal-Elkin                        Elkin-Elkin             NC-Elkin             USA-Elkin
3372        ZER-Pottsville  Schuylkill Cty/Joe Zerbey-Pottsville              Pottsville-Pottsville        PA-Pottsville        USA-Pottsville
3373       ZPH-Zephyrhills     Zephyrhills Municipal-Zephyrhills            Zephyrhills-Zephyrhills       FL-Zephyrhills       USA-Zephyrhills
3374              ZUN-Zuni                       Black Rock-Zuni                          Zuni-Zuni              NM-Zuni              USA-Zuni
3375        ZZV-Zanesville       Zanesville Municipal-Zanesville              Zanesville-Zanesville        OH-Zanesville        USA-Zanesville

[3376 rows x 5 columns]
```

```python
df[df.STR.contains('Cloud').any(axis=1)].STR.replace(pat='Cloud', repl='Butt')
```
```
     iata                name         city state country
386   42C          White Butt   White Butt    MI     USA
653   7V7  Red Butt Municipal     Red Butt    NE     USA
1496  FCM         Flying Butt  Minneapolis    MN     USA
3015  STC    St Butt Regional      St Butt    MN     USA
```

```python
df.STR.split()
```
```
       iata                           name                 city state country
0     [00M]                      [Thigpen]       [Bay, Springs]  [MS]   [USA]
1     [00R]        [Livingston, Municipal]         [Livingston]  [TX]   [USA]
2     [00V]                 [Meadow, Lake]  [Colorado, Springs]  [CO]   [USA]
3     [01G]                 [Perry-Warsaw]              [Perry]  [NY]   [USA]
4     [01J]            [Hilliard, Airpark]           [Hilliard]  [FL]   [USA]
...     ...                            ...                  ...   ...     ...
3371  [ZEF]             [Elkin, Municipal]              [Elkin]  [NC]   [USA]
3372  [ZER]  [Schuylkill, Cty/Joe, Zerbey]         [Pottsville]  [PA]   [USA]
3373  [ZPH]       [Zephyrhills, Municipal]        [Zephyrhills]  [FL]   [USA]
3374  [ZUN]                  [Black, Rock]               [Zuni]  [NM]   [USA]
3375  [ZZV]        [Zanesville, Municipal]         [Zanesville]  [OH]   [USA]

[3376 rows x 5 columns]
```

```python
df.STR.split(expand=True)
```
```
     iata_0        name_0     name_1  name_2 name_3 name_4 name_5 name_6       city_0   city_1 city_2 city_3 state_0 country_0 country_1 country_2 country_3
0       00M       Thigpen       None    None   None   None   None   None          Bay  Springs   None   None      MS       USA      None      None      None
1       00R    Livingston  Municipal    None   None   None   None   None   Livingston     None   None   None      TX       USA      None      None      None
2       00V        Meadow       Lake    None   None   None   None   None     Colorado  Springs   None   None      CO       USA      None      None      None
3       01G  Perry-Warsaw       None    None   None   None   None   None        Perry     None   None   None      NY       USA      None      None      None
4       01J      Hilliard    Airpark    None   None   None   None   None     Hilliard     None   None   None      FL       USA      None      None      None
...     ...           ...        ...     ...    ...    ...    ...    ...          ...      ...    ...    ...     ...       ...       ...       ...       ...
3371    ZEF         Elkin  Municipal    None   None   None   None   None        Elkin     None   None   None      NC       USA      None      None      None
3372    ZER    Schuylkill    Cty/Joe  Zerbey   None   None   None   None   Pottsville     None   None   None      PA       USA      None      None      None
3373    ZPH   Zephyrhills  Municipal    None   None   None   None   None  Zephyrhills     None   None   None      FL       USA      None      None      None
3374    ZUN         Black       Rock    None   None   None   None   None         Zuni     None   None   None      NM       USA      None      None      None
3375    ZZV    Zanesville  Municipal    None   None   None   None   None   Zanesville     None   None   None      OH       USA      None      None      None

[3376 rows x 17 columns]
```

```python
data = [df.name[i*6:(i+1)*(6)].tolist() for i in range(30)]
idx = pandas.MultiIndex.from_product((['A','B','C'], ['foo','bar'], ['one','two','three','four','five']))
cols = pandas.MultiIndex.from_product((['foo','bar','baz'], ['one','two']))
DF = pandas.DataFrame(data=data, index=idx, columns=cols)
DF.STR.split('-', expand=True)
```
```
                                     foo                                                                                                bar                                                                           baz
                                   one_0            one_1                           two_0            two_1                            one_0        one_1                       two_0   two_1                        one_0         one_1                   two_0            two_1       two_2
A foo one                        Thigpen             None            Livingston Municipal             None                      Meadow Lake         None                       Perry  Warsaw             Hilliard Airpark          None       Tishomingo County             None        None
      two                          Gragg             Wade                         Capitol             None                Columbiana County         None            Memphis Memorial    None               Calhoun County          None        Hawley Municipal             None        None
      three                     Griffith     Merrillville                     Gatesville       City/County                           Eureka         None            Moton  Municipal    None                   Schaumburg          None         Rolla Municipal             None        None
      four              Eupora Municipal             None                         Randall             None                   Jackpot/Hayden         None               Dekalb County    None         Gladewater Municipal          None           Fitch H Beach             None        None
      five        Central City Municipal             None              Wetumpka Municipal             None                Stanley Municipal         None               Harvard State    None                     Carthage  Leake County                  Butler   Choctaw County        None
  bar one                  Jekyll Island             None               Sargent Municipal             None             Charleston Municipal         None        South Capitol Street    None         Smithville Municipal          None             Bibb County             None        None
      two         Elizabethton Municipal             None                   Pilot Station             None                        Col. Dyke         None        Hartington Municipal    None                Turners Falls          None                  Warren       Sugar Bush        None
      three                    Elizabeth             None                            Dacy             None                 Pender Municipal         None       South Haven Municipal    None         Gettysburg Municipal          None                Moriarty             None        None
      four                    Crownpoint             None                 Bowie Municipal             None              Loup City Municipal         None  Fountainhead Lodge Airpark    None    William R Pogue Municipal          None      Tishomingo Airpark             None        None
      five        North Buffalo Suburban             None              Tecumseh Municipal             None                  Williams County         None       Finger Lakes Regional    None               Trego Wakeeney          None               Cynthiana  Harrison County        None
B foo one            Abbeville Municipal             None               Florala Municipal             None               Headland Municipal         None          Humboldt Municipal    None                    Goldfield          None                    Jean             None        None
      two                       Echo Bay             None                 Dumas Municipal             None                            Scott         None               Benton County    None             Humphreys County          None           Panola County             None        None
      three                      Byerley             None                    Calaveras Co  Maury Rasmussen                Corning Municipal         None                  University    None                 Shelter Cove          None             Shingletown             None        None
      four                      Columbia    Marion County                Atmore Municipal             None  Abbeville Chris Crusta Memorial         None            Concordia Parish    None                David G Joyce          None               Red River             None        None
      five                 Dorothy Scott             None  Jefferson County International             None                Harriet Alexander         None             Pioneer Village    None    Brookneal/Campbell County          None           Mission Sioux             None        None
  bar one                        Kayenta             None                            Galt             None                Winsted Municipal         None               Holmes County    None                     Wallkill          None                  Owyhee             None        None
      two              Clayton Municipal             None                     Clarion Cty             None              Schaumburg Heliport         None                Early County    None            Brenham Municipal          None      Rochelle Municipal             None        None
      three              Tower Municipal             None               Brewton Municipal             None               Superior Municipal         None          Le Sueur Municipal    None                     Lakeview          None        Eureka Municipal             None        None
      four                        Trinca             None                     Carl Folsom             None             Hollandale Municipal         None                  Todd Field    None            Haskell Municipal          None             Cook County             None        None
      five                          Luka             None                      McCarthy 2             None                      Nunapitchuk         None               Seneca County    None             Dawson Municipal          None  Myrtle Creek Municipal             None        None
C foo one                   Port Bucyrus  Crawford County         Donalsonville Municipal             None                       Boise City         None             Magee Municipal    None                   Cross Keys          None               Manokotak             None        None
      two                Franklin County             None                 McCreary County             None                   Jackson County         None                   C A Moore    None                       Camden          None     Port Protection SPB             None        None
      three              Martin Campbell             None                    Macon County             None                      Middlesboro  Bell County              Jackson County    None               Autauga County          None         Dexter Regional             None        None
      four                  Columbia Cty             None                      Fair Haven             None              Mansfield Municipal         None                        Clow    None            Milbank Municipal          None                 Canton         Plymouth     Mettetal
      five              Platte Municipal             None                Hector Municipal             None                Webster Municipal         None          Redfield Municipal    None             Downtown Ardmore          None  Lake Murray State Park             None        None
  bar one               Madill Municipal             None            Bridgeport Municipal             None                      Wood County         None       Kent State University    None            Grand Canyon West          None                 Freedom             None        None
      two                        Michael             None                     Creve Coeur             None        Effingham County Memorial         None    Linn State Tech. College    None              Casey Municipal          None                Freehold             None        None
      three             Delphi Municipal             None                             Tri           County                Lindsay Municipal         None              David J. Perry    None            Waynoka Municipal          None       Satanta Municipal             None        None
      four   St. John the Baptist Parish             None                      Lincoln Co             None              Escalante Municipal         None                     Parowan    None  North Little Rock Municipal          None       Belzoni Municipal             None        None
      five                         Posey             None              Portland Municipal             None                           Fulton         None     Mountain Grove Memorial    None                      Spadaro          None           Woodbine Muni             None        None
```

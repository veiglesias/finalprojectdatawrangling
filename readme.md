# Demonstrating Quarto for GitHub READMEs


``` python
# First 10 rows, info, and data types for nat_cor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objects as go

nat_cor = pd.read_csv("nationalism_corruption.csv")
v_dem = pd.read_csv(("V-Dem-CY-Full.csv"), low_memory=False) # to ensure proper data 

nat_cor.head(10)
nat_cor.info()
nat_cor.dtypes

# Summary statistics
nat_cor.describe()

# Check for missing values and duplicates
nat_cor.isnull().sum()
nat_cor.duplicated().sum()

# The data covers a lot of years so I will cut down to the most recent decade for both datasets
filtered_nat_cor = nat_cor[(nat_cor['year'] >= 2012) & (nat_cor['year'] <= 2022)]
filtered_nat_cor

# There's only 30 columns but I will filter which columns/variables I will keep for the analysis
col_f_nat_cor = [
    'country', 'year', 'top_ideology', 'v2exl_legitideol', 'wdi_population'
] #columns of interest from this dataset:
# top_ideology: Top ideology in the country
# v2exl_legitideol: Legitimacy of the electoral process
# wdi_population: Population (World Bank)

new_nat_cor = filtered_nat_cor[col_f_nat_cor].copy()
new_nat_cor # filtered dataset with columns of interest
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18628 entries, 0 to 18627
    Data columns (total 30 columns):
     #   Column                                                      Non-Null Count  Dtype  
    ---  ------                                                      --------------  -----  
     0   iso3c                                                       18628 non-null  object 
     1   year                                                        18628 non-null  int64  
     2   country_name                                                18628 non-null  object 
     3   v2exl_legitideol                                            18293 non-null  float64
     4   v2exl_legitideolcr_0                                        18506 non-null  float64
     5   v2exl_legitideolcr_1                                        18506 non-null  float64
     6   v2exl_legitideolcr_2                                        18506 non-null  float64
     7   v2exl_legitideolcr_3                                        18506 non-null  float64
     8   v2exl_legitideolcr_4                                        18506 non-null  float64
     9   top_ideology                                                18506 non-null  object 
     10  nationalism                                                 17551 non-null  float64
     11  country                                                     10211 non-null  object 
     12  iso2c                                                       10148 non-null  object 
     13  wdi_prop_less_2_usd_day                                     1926 non-null   float64
     14  wdi_gdppc_nominal                                           8963 non-null   float64
     15  wdi_gdppc_ppp                                               5354 non-null   float64
     16  wdi_urban_population_pct                                    10211 non-null  float64
     17  wdi_urban_pop_1m_cities_pct                                 7327 non-null   float64
     18  wdi_gini_index                                              1911 non-null   float64
     19  wdi_life_expectancy_at_birth                                9803 non-null   float64
     20  wdi_pop_over_65                                             10211 non-null  float64
     21  wdi_pop_under_15                                            10211 non-null  float64
     22  wdi_population                                              10211 non-null  float64
     23  corruption                                                  1884 non-null   float64
     24  regime_use_of_ideology_measure                              18293 non-null  float64
     25  ideology_is_nationalist_percent_of_experts                  18506 non-null  float64
     26  ideology_is_socialist_or_communist_percent_of_experts       18506 non-null  float64
     27  ideology_is_restorative_or_conservative_percent_of_experts  18506 non-null  float64
     28  ideology_is_separatist_or_autonomist_percent_of_experts     18506 non-null  float64
     29  ideology_is_religious_percent_of_experts                    18506 non-null  float64
    dtypes: float64(24), int64(1), object(5)
    memory usage: 4.3+ MB

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|       | country     | year | top_ideology | v2exl_legitideol | wdi_population |
|-------|-------------|------|--------------|------------------|----------------|
| 111   | Afghanistan | 2012 | nationalist  | -0.899           | 30466479.0     |
| 112   | Afghanistan | 2013 | nationalist  | -0.899           | 31541209.0     |
| 113   | Afghanistan | 2014 | nationalist  | -0.701           | 32716210.0     |
| 114   | Afghanistan | 2015 | nationalist  | -0.701           | 33753499.0     |
| 115   | Afghanistan | 2016 | nationalist  | -0.701           | 34636207.0     |
| ...   | ...         | ...  | ...          | ...              | ...            |
| 18623 | Zimbabwe    | 2018 | nationalist  | 0.459            | 15052184.0     |
| 18624 | Zimbabwe    | 2019 | nationalist  | 0.459            | 15354608.0     |
| 18625 | Zimbabwe    | 2020 | nationalist  | 0.408            | 15669666.0     |
| 18626 | Zimbabwe    | 2021 | nationalist  | 1.313            | 15993524.0     |
| 18627 | Zimbabwe    | 2022 | nationalist  | 1.313            | 15993524.0     |

<p>1914 rows × 5 columns</p>
</div>

``` python
# First 10 rows, info, and data types for v_dem
v_dem.head(10)
v_dem.info()
v_dem.dtypes

# Summary statistics
v_dem.describe()

# Check for missing values and duplicates
v_dem.isnull().sum()
v_dem.duplicated().sum()

# The data covers a lot of years so I will also cut down to the most recent decade for both datasets
filtered_v_dem = v_dem[(v_dem['year'] >= 2012) & (v_dem['year'] <= 2022)].copy()

# Considering there's over 4,000 columns (4607!), I will filter which columns/variables I will keep for the analysis
col_f_v_dem = [
    'country_name', 'year', 'v2x_polyarchy', 'v2mecenefm', 'v2x_corr', 'e_wbgi_gee', 'e_wbgi_vae', 'v2x_cspart'
] #columns of interest from this dataset:
# v2x_polyarchy = Electoral Democracy Index (0-1 scale)
# v2mecenefm = Mean Electoral Competitiveness Index (0-1 scale)
# v2x_corr = Corruption Index (0-1 scale)
# e_wbgi_gee = World Bank Governance Indicator - Government Effectiveness (-2.5 to 2.5 scale)
# e_wbgi_vae = World Bank Governance Indicator - Voice and Accountability (-2.5 to 2.5 scale)
# v2x_cspart = Civil Society Participation Index (0-1 scale)

new_v_dem = filtered_v_dem[col_f_v_dem].copy()
new_v_dem = new_v_dem.rename(columns={'country_name': 'country'})

new_v_dem # filtered and renamed dataset with columns of interest
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27913 entries, 0 to 27912
    Columns: 4607 entries, country_name to e_pt_coup_attempts
    dtypes: float64(4560), int64(18), object(29)
    memory usage: 981.1+ MB

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | country | year | v2x_polyarchy | v2mecenefm | v2x_corr | e_wbgi_gee | e_wbgi_vae | v2x_cspart |
|----|----|----|----|----|----|----|----|----|
| 223 | Mexico | 2012 | 0.649 | 0.523 | 0.661 | 0.296 | 0.113 | 0.698 |
| 224 | Mexico | 2013 | 0.623 | 0.969 | 0.759 | 0.303 | 0.108 | 0.632 |
| 225 | Mexico | 2014 | 0.620 | 0.836 | 0.759 | 0.134 | 0.000 | 0.615 |
| 226 | Mexico | 2015 | 0.639 | 1.101 | 0.726 | 0.118 | -0.075 | 0.625 |
| 227 | Mexico | 2016 | 0.640 | 1.101 | 0.726 | 0.028 | -0.068 | 0.617 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 26507 | Zanzibar | 2018 | 0.268 | -0.959 | 0.659 | NaN | NaN | 0.683 |
| 26508 | Zanzibar | 2019 | 0.266 | -0.626 | 0.695 | NaN | NaN | 0.641 |
| 26509 | Zanzibar | 2020 | 0.271 | -0.793 | 0.693 | NaN | NaN | 0.632 |
| 26510 | Zanzibar | 2021 | 0.285 | -0.793 | 0.707 | NaN | NaN | 0.647 |
| 26511 | Zanzibar | 2022 | 0.294 | 0.175 | 0.657 | NaN | NaN | 0.641 |

<p>1969 rows × 8 columns</p>
</div>

``` python
combined = pd.merge(new_nat_cor, new_v_dem, left_on=['country', 'year'], right_on=['country', 'year'], how='inner')
combined # merged dataset
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | country | year | top_ideology | v2exl_legitideol | wdi_population | v2x_polyarchy | v2mecenefm | v2x_corr | e_wbgi_gee | e_wbgi_vae | v2x_cspart |
|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | Afghanistan | 2012 | nationalist | -0.899 | 30466479.0 | 0.362 | 0.728 | 0.961 | -1.376 | -1.267 | 0.671 |
| 1 | Afghanistan | 2013 | nationalist | -0.899 | 31541209.0 | 0.357 | 0.764 | 0.941 | -1.399 | -1.240 | 0.676 |
| 2 | Afghanistan | 2014 | nationalist | -0.701 | 32716210.0 | 0.355 | 0.764 | 0.927 | -1.359 | -1.135 | 0.635 |
| 3 | Afghanistan | 2015 | nationalist | -0.701 | 33753499.0 | 0.353 | 0.764 | 0.904 | -1.396 | -1.118 | 0.656 |
| 4 | Afghanistan | 2016 | nationalist | -0.701 | 34636207.0 | 0.340 | 0.599 | 0.905 | -1.290 | -1.038 | 0.620 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1678 | Zimbabwe | 2018 | nationalist | 0.459 | 15052184.0 | 0.303 | -0.435 | 0.762 | -1.298 | -1.137 | 0.815 |
| 1679 | Zimbabwe | 2019 | nationalist | 0.459 | 15354608.0 | 0.292 | -0.570 | 0.765 | -1.320 | -1.164 | 0.799 |
| 1680 | Zimbabwe | 2020 | nationalist | 0.408 | 15669666.0 | 0.294 | -0.392 | 0.795 | -1.356 | -1.113 | 0.790 |
| 1681 | Zimbabwe | 2021 | nationalist | 1.313 | 15993524.0 | 0.289 | -0.352 | 0.826 | -1.305 | -1.136 | 0.807 |
| 1682 | Zimbabwe | 2022 | nationalist | 1.313 | 15993524.0 | 0.286 | -0.575 | 0.806 | -1.255 | -1.102 | 0.786 |

<p>1683 rows × 11 columns</p>
</div>

``` python
# Creating a regional classification for countries
sub_region_class = pd.Series(index=combined.index, dtype=str)

# North America
sub_region_class.loc[combined['country'].isin(['United States of America', 'Mexico', 'Canada'])] = 'North America'
# Central America
sub_region_class.loc[combined['country'].isin(['El Salvador', 'Guatemala', 'Honduras', 'Belize', 'Nicaragua', 'Costa Rica', 'Panama'])] = 'Central America'
# South America
sub_region_class.loc[combined['country'].isin(['Brazil', 'Argentina', 'Chile', 'Ecuador', 'Bolivia', 'Guyana', 'Paraguay', 'Colombia', 'Peru', 'Venezuela', 'Suriname', 'Uruguay'])] = 'South America'
# Caribbean
sub_region_class.loc[combined['country'].isin(['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Cuba', 'Dominica', 'Dominican Republic', 'Grenada', 'Haiti', 'Jamaica', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago'])] = 'Caribbean'
# North Europe
sub_region_class.loc[combined['country'].isin(['Norway', 'Sweden', 'Denmark', 'Finland', 'Iceland', 'Estonia', 'Latvia', 'Lithuania', 'Ireland', 'United Kingdom'])] = 'North Europe'
# West Europe
sub_region_class.loc[combined['country'].isin(['Belgium', 'France', 'Luxembourg', 'Netherlands'])] = 'West Europe'
# Central Europe
sub_region_class.loc[combined['country'].isin(['Austria', 'Czechia', 'Germany', 'Hungary', 'Liechtenstein', 'Slovakia', 'Poland', 'Slovenia', 'Switzerland'])] = 'Central Europe'
# East Europe
sub_region_class.loc[combined['country'].isin(['Russia', 'Romania', 'Bulgaria', 'Ukraine', 'Belarus', 'Moldova', 'Georgia'])] = 'East Europe'
# South Europe
sub_region_class.loc[combined['country'].isin(['Spain', 'Portugal', 'Albania', 'Andorra', 'Bosnia and Herzegovina', 'Crotia', 'Cyprus', 'Greece', 'Italy', 'Kosovo', 'Malta', 'Montenegro', 'North Macedonia', 'San Marino', 'Serbia', 'Vatican City', 'Croatia'])] = 'South Europe'
# Middle East
sub_region_class.loc[combined['country'].isin(['Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Turkey', 'United Arab Emirates', 'Yemen', 'Armenia', 'Azerbaijan'])] = 'Middle East'
# East Asia
sub_region_class.loc[combined['country'].isin(['China', 'Japan', 'North Korea', 'South Korea', 'Taiwan', 'Mongolia', 'Hong Kong'])] = 'East Asia'
# South Asia
sub_region_class.loc[combined['country'].isin(['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka'])] = 'South Asia'
# Central Asia
sub_region_class.loc[combined['country'].isin(['Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan'])] = 'Central Asia'
# Southeast Asia
sub_region_class.loc[combined['country'].isin(['Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Burma/Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'])] = 'Southeast Asia'
# North Africa
sub_region_class.loc[combined['country'].isin(['Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia'])] = 'North Africa'
# West Africa
sub_region_class.loc[combined['country'].isin(['Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Togo', 'The Gambia'])] = 'West Africa'
# Central Africa
sub_region_class.loc[combined['country'].isin(['Cameroon', 'Central African Republic', 'Chad', 'Republic of the Congo', 'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 'Sao Tome and Principe'])] = 'Central Africa'
# East Africa
sub_region_class.loc[combined['country'].isin(['Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya', 'Madagascar', 'Mauritius', 'Rwanda', 'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda'])] = 'East Africa'
# Southern Africa
sub_region_class.loc[combined['country'].isin(['Angola', 'Botswana', 'Eswatini', 'Lesotho', 'Malawi', 'Mozambique', 'Namibia', 'South Africa', 'Zambia', 'Zimbabwe'])] = 'Southern Africa'
# Oceania
sub_region_class.loc[combined['country'].isin(['Australia', 'New Zealand', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'])] = 'Oceania'

combined['sub_region_class'] = sub_region_class
display(combined.tail(150))

combined['sub_region_class'].isnull().sum() # verifying every country in the data has a sub-region assigned
sub_region_class.unique() # sub-region classification check
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | country | year | top_ideology | v2exl_legitideol | wdi_population | v2x_polyarchy | v2mecenefm | v2x_corr | e_wbgi_gee | e_wbgi_vae | v2x_cspart | sub_region_class |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1533 | Turkmenistan | 2016 | nationalist | 2.634 | 5868561.0 | 0.144 | -2.727 | 0.931 | -0.914 | -2.171 | 0.032 | Central Asia |
| 1534 | Turkmenistan | 2017 | nationalist | 2.634 | 5968383.0 | 0.147 | -2.727 | 0.911 | -1.012 | -2.160 | 0.054 | Central Asia |
| 1535 | Turkmenistan | 2018 | nationalist | 2.634 | 6065066.0 | 0.149 | -2.727 | 0.911 | -1.090 | -2.153 | 0.054 | Central Asia |
| 1536 | Turkmenistan | 2019 | nationalist | 2.087 | 6158420.0 | 0.149 | -2.727 | 0.913 | -1.018 | -2.184 | 0.053 | Central Asia |
| 1537 | Turkmenistan | 2020 | nationalist | 2.087 | 6250438.0 | 0.150 | -2.727 | 0.909 | -1.023 | -2.028 | 0.053 | Central Asia |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1678 | Zimbabwe | 2018 | nationalist | 0.459 | 15052184.0 | 0.303 | -0.435 | 0.762 | -1.298 | -1.137 | 0.815 | Southern Africa |
| 1679 | Zimbabwe | 2019 | nationalist | 0.459 | 15354608.0 | 0.292 | -0.570 | 0.765 | -1.320 | -1.164 | 0.799 | Southern Africa |
| 1680 | Zimbabwe | 2020 | nationalist | 0.408 | 15669666.0 | 0.294 | -0.392 | 0.795 | -1.356 | -1.113 | 0.790 | Southern Africa |
| 1681 | Zimbabwe | 2021 | nationalist | 1.313 | 15993524.0 | 0.289 | -0.352 | 0.826 | -1.305 | -1.136 | 0.807 | Southern Africa |
| 1682 | Zimbabwe | 2022 | nationalist | 1.313 | 15993524.0 | 0.286 | -0.575 | 0.806 | -1.255 | -1.102 | 0.786 | Southern Africa |

<p>150 rows × 12 columns</p>
</div>

    array(['South Asia', 'Southern Africa', 'South Europe', 'Middle East',
           'South America', 'Oceania', 'Central Europe', 'East Africa',
           'West Europe', 'West Africa', 'East Europe', 'Caribbean',
           'Central Africa', 'North America', 'East Asia', 'Central America',
           'North Europe', 'North Africa', 'Southeast Asia', 'Central Asia'],
          dtype=object)

``` python
# Creating a regional classification for countries
region_class = pd.Series(index=combined.index, dtype=str)

# America
region_class.loc[combined['country'].isin(['United States of America', 'Mexico', 'Canada', 'El Salvador', 'Guatemala', 'Honduras', 'Belize', 'Nicaragua', 'Costa Rica', 'Panama', 'Brazil', 'Argentina', 'Chile', 'Ecuador', 'Bolivia', 'Guyana', 'Paraguay', 'Colombia', 'Peru', 'Venezuela', 'Suriname', 'Uruguay', 'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Cuba', 'Dominica', 'Dominican Republic', 'Grenada', 'Haiti', 'Jamaica', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago'])] = 'America'

# Europe
region_class.loc[combined['country'].isin(['Norway', 'Sweden', 'Denmark', 'Finland', 'Iceland', 'Estonia', 'Latvia', 'Lithuania', 'Ireland', 'United Kingdom', 'Belgium', 'France', 'Luxembourg', 'Netherlands', 'Austria', 'Czechia', 'Germany', 'Hungary', 'Liechtenstein', 'Slovakia', 'Poland', 'Slovenia', 'Switzerland', 'Russia', 'Romania', 'Bulgaria', 'Ukraine', 'Belarus', 'Moldova', 'Georgia', 'Spain', 'Portugal', 'Albania', 'Andorra', 'Bosnia and Herzegovina', 'Crotia', 'Cyprus', 'Greece', 'Italy', 'Kosovo', 'Malta', 'Montenegro', 'North Macedonia', 'San Marino', 'Serbia', 'Vatican City', 'Croatia'])] = 'Europe'

# Middle East
region_class.loc[combined['country'].isin(['Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Turkey', 'United Arab Emirates', 'Yemen', 'Armenia', 'Azerbaijan'])] = 'Middle East'

# Asia
region_class.loc[combined['country'].isin(['China', 'Japan', 'North Korea', 'South Korea', 'Taiwan', 'Mongolia', 'Hong Kong', 'Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka', 'Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Burma/Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'])] = 'Asia'

# Africa
region_class.loc[combined['country'].isin(['Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Togo', 'The Gambia', 'Cameroon', 'Central African Republic', 'Chad', 'Republic of the Congo', 'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 'Sao Tome and Principe', 'Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya', 'Madagascar', 'Mauritius', 'Rwanda', 'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda', 'Angola', 'Botswana', 'Eswatini', 'Lesotho', 'Malawi', 'Mozambique', 'Namibia', 'South Africa', 'Zambia', 'Zimbabwe'])] = 'Africa'

# Oceania
region_class.loc[combined['country'].isin(['Australia', 'New Zealand', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'])] = 'Oceania'

combined['region_class'] = region_class
display(combined.tail(150))

combined['region_class'].isnull().sum() # verifying every country in the data has a region assigned
region_class.unique() # region classification check
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | country | year | top_ideology | v2exl_legitideol | wdi_population | v2x_polyarchy | v2mecenefm | v2x_corr | e_wbgi_gee | e_wbgi_vae | v2x_cspart | sub_region_class | region_class |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1533 | Turkmenistan | 2016 | nationalist | 2.634 | 5868561.0 | 0.144 | -2.727 | 0.931 | -0.914 | -2.171 | 0.032 | Central Asia | Asia |
| 1534 | Turkmenistan | 2017 | nationalist | 2.634 | 5968383.0 | 0.147 | -2.727 | 0.911 | -1.012 | -2.160 | 0.054 | Central Asia | Asia |
| 1535 | Turkmenistan | 2018 | nationalist | 2.634 | 6065066.0 | 0.149 | -2.727 | 0.911 | -1.090 | -2.153 | 0.054 | Central Asia | Asia |
| 1536 | Turkmenistan | 2019 | nationalist | 2.087 | 6158420.0 | 0.149 | -2.727 | 0.913 | -1.018 | -2.184 | 0.053 | Central Asia | Asia |
| 1537 | Turkmenistan | 2020 | nationalist | 2.087 | 6250438.0 | 0.150 | -2.727 | 0.909 | -1.023 | -2.028 | 0.053 | Central Asia | Asia |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1678 | Zimbabwe | 2018 | nationalist | 0.459 | 15052184.0 | 0.303 | -0.435 | 0.762 | -1.298 | -1.137 | 0.815 | Southern Africa | Africa |
| 1679 | Zimbabwe | 2019 | nationalist | 0.459 | 15354608.0 | 0.292 | -0.570 | 0.765 | -1.320 | -1.164 | 0.799 | Southern Africa | Africa |
| 1680 | Zimbabwe | 2020 | nationalist | 0.408 | 15669666.0 | 0.294 | -0.392 | 0.795 | -1.356 | -1.113 | 0.790 | Southern Africa | Africa |
| 1681 | Zimbabwe | 2021 | nationalist | 1.313 | 15993524.0 | 0.289 | -0.352 | 0.826 | -1.305 | -1.136 | 0.807 | Southern Africa | Africa |
| 1682 | Zimbabwe | 2022 | nationalist | 1.313 | 15993524.0 | 0.286 | -0.575 | 0.806 | -1.255 | -1.102 | 0.786 | Southern Africa | Africa |

<p>150 rows × 13 columns</p>
</div>

    array(['Asia', 'Africa', 'Europe', 'Middle East', 'America', 'Oceania'],
          dtype=object)

``` python
# Because the corruption data is only available from 2012 onwards, I will focus on the extracting a good analysis from the years 2012 to 2022. This will ensure that the analysis is based on complete data for all variables of interest. It's worth noting that this may result in a smaller dataset but as long as I recognize the limitation of the data and timeframe and choose the variables for analysis carefully, I can still derive meaningful insights.

# Calculate the change for each country across the entire time period (2012-2022), using the last and first year values in a new DataFrame
# Using .apply() because I need to calculate multiple aggregated values per group (polyarchy, wbgi_gee, v2x_corr)
country_performance = combined.groupby('country', as_index=False).apply(lambda x: pd.Series({
    'polyarchy_change': x['v2x_polyarchy'].iloc[-1] - x['v2x_polyarchy'].iloc[0],
    'e_wbgi_gee_change': (x['e_wbgi_gee'].iloc[-1] - x['e_wbgi_gee'].iloc[0]) / x['e_wbgi_gee'].iloc[0] if (x['e_wbgi_gee'].iloc[0] != 0 and not pd.isna(x['e_wbgi_gee'].iloc[0])) else np.nan,
    'v2x_corr_change': (x['v2x_corr'].iloc[-1] - x['v2x_corr'].iloc[0]) / x['v2x_corr'].iloc[0] if (x['v2x_corr'].iloc[0] != 0 and not pd.isna(x['v2x_corr'].iloc[0])) else np.nan,
    'v2x_polyarchy_last_year': x['v2x_polyarchy'].iloc[-1] # Keep the last year's polyarchy score for context
}), include_groups=False)

# 'country' as the index for country_performance
country_performance = country_performance.set_index('country')

change_variables = {
    'polyarchy_change': 'Electoral Democracy (Polyarchy) Index',
    'e_wbgi_gee_change': 'Government Effectiveness Index',
    'v2x_corr_change': 'Corruption Index'
}

for variable, title in change_variables.items():
    print(f"\n--- Top 10 Improvers based on {title} ---")
    top_improvers = country_performance.sort_values(by=variable, ascending=False).head(10)
    print(top_improvers[[variable]])

    print(f"\n--- Top 10 Decliners based on {title} ---")
    worst_decliners = country_performance.sort_values(by=variable, ascending=True).head(10)
    print(worst_decliners[[variable]])
```


    --- Top 10 Improvers based on Electoral Democracy (Polyarchy) Index ---
                     polyarchy_change
    country                          
    Armenia                     0.298
    Seychelles                  0.277
    Madagascar                  0.271
    Fiji                        0.262
    Nepal                       0.239
    Sri Lanka                   0.211
    Malaysia                    0.130
    Honduras                    0.118
    Kenya                       0.116
    North Macedonia             0.107

    --- Top 10 Decliners based on Electoral Democracy (Polyarchy) Index ---
                  polyarchy_change
    country                       
    Tunisia                 -0.484
    Hungary                 -0.325
    Poland                  -0.318
    Burkina Faso            -0.295
    Thailand                -0.289
    Afghanistan             -0.286
    India                   -0.272
    Mauritius               -0.251
    El Salvador             -0.243
    Serbia                  -0.236

    --- Top 10 Improvers based on Government Effectiveness Index ---
                  e_wbgi_gee_change
    country                        
    Suriname              71.615385
    China                 32.000000
    Saudi Arabia          29.684211
    Jamaica               12.130435
    Armenia                8.812500
    Bulgaria               8.678571
    Lebanon                5.079167
    Tunisia                4.750000
    Jordan                 3.875000
    Brazil                 2.652174

    --- Top 10 Decliners based on Government Effectiveness Index ---
                  e_wbgi_gee_change
    country                        
    Colombia             -15.000000
    India                 -3.371795
    Rwanda                -3.274510
    Indonesia             -2.324242
    Kuwait                -2.258427
    Mexico                -1.949324
    South Africa          -1.735294
    Vietnam               -1.705645
    Fiji                  -1.564347
    Panama                -1.396875

    --- Top 10 Improvers based on Corruption Index ---
              v2x_corr_change
    country                  
    Sweden           1.500000
    Uruguay          0.976190
    Botswana         0.883562
    Norway           0.833333
    Zambia           0.800000
    Austria          0.790323
    Namibia          0.747368
    Iceland          0.692308
    Poland           0.630952
    Georgia          0.567010

    --- Top 10 Decliners based on Corruption Index ---
                         v2x_corr_change
    country                             
    Seychelles                 -0.737255
    Benin                      -0.698517
    Moldova                    -0.677515
    Armenia                    -0.623086
    Trinidad and Tobago        -0.472050
    Tanzania                   -0.463830
    Romania                    -0.463744
    Japan                      -0.431193
    Sierra Leone               -0.419876
    Afghanistan                -0.409990

``` python
# 1. How does the level of corruption vary across world regions?
plt.figure(figsize=(14, 8))
sns.boxplot(x='region_class', y='v2x_corr', data=combined, hue='region_class', legend=False)
plt.title('Corruption Index Distribution by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Corruption Index', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-8-output-1.png)

``` python
# 2. Which political and institutional varaibles contributed most to the change in the corruption index in Central America between 2012 and 2022?

ca_data = combined[combined['sub_region_class'] == 'Central America'].copy()
display(ca_data.head())
display(ca_data.tail())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | country | year | top_ideology | v2exl_legitideol | wdi_population | v2x_polyarchy | v2mecenefm | v2x_corr | e_wbgi_gee | e_wbgi_vae | v2x_cspart | sub_region_class | region_class |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 341 | Costa Rica | 2012 | nationalist | -2.243 | 4736593.0 | 0.909 | 2.391 | 0.187 | 0.474 | 1.088 | 0.949 | Central America | America |
| 342 | Costa Rica | 2013 | nationalist | -2.243 | 4791535.0 | 0.908 | 2.350 | 0.187 | 0.457 | 1.078 | 0.946 | Central America | America |
| 343 | Costa Rica | 2014 | nationalist | -1.974 | 4844288.0 | 0.905 | 1.813 | 0.179 | 0.363 | 1.138 | 0.946 | Central America | America |
| 344 | Costa Rica | 2015 | nationalist | -1.974 | 4895242.0 | 0.905 | 1.813 | 0.182 | 0.307 | 1.152 | 0.916 | Central America | America |
| 345 | Costa Rica | 2016 | nationalist | -1.974 | 4945205.0 | 0.904 | 2.160 | 0.187 | 0.284 | 1.118 | 0.916 | Central America | America |

</div>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | country | year | top_ideology | v2exl_legitideol | wdi_population | v2x_polyarchy | v2mecenefm | v2x_corr | e_wbgi_gee | e_wbgi_vae | v2x_cspart | sub_region_class | region_class |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1381 | El Salvador | 2018 | socialist/communist | -0.905 | 6276342.0 | 0.635 | 0.844 | 0.685 | -0.490 | 0.030 | 0.749 | Central America | America |
| 1382 | El Salvador | 2019 | nationalist | -1.230 | 6280217.0 | 0.594 | -0.134 | 0.598 | -0.531 | 0.118 | 0.590 | Central America | America |
| 1383 | El Salvador | 2020 | nationalist | -1.230 | 6292731.0 | 0.522 | -0.874 | 0.668 | -0.261 | 0.071 | 0.532 | Central America | America |
| 1384 | El Salvador | 2021 | nationalist | -1.230 | 6314167.0 | 0.409 | -1.433 | 0.711 | -0.341 | -0.060 | 0.444 | Central America | America |
| 1385 | El Salvador | 2022 | nationalist | -1.230 | 6314167.0 | 0.381 | -1.998 | 0.747 | -0.320 | -0.372 | 0.428 | Central America | America |

</div>

``` python
variables_to_analyze = [
    'v2x_corr', 'v2mecenefm', 'v2exl_legitideol', 'wdi_population',
    'v2x_polyarchy', 'e_wbgi_gee', 'e_wbgi_vae', 'v2x_cspart'
]

# Calculate delta
ca_changes = ca_data.groupby('country').apply(lambda x: pd.Series({
    f'{col}_change': x[col].iloc[-1] - x[col].iloc[0]
    for col in variables_to_analyze
}), include_groups=False)

# Calculate the mean of these changes across all countries in Central America
avg_ca_changes = ca_changes.mean()

print("Average Changes in Central America (2022 from 2012):")
print(avg_ca_changes)
```

    Average Changes in Central America (2022 from 2012):
    v2x_corr_change                -0.011000
    v2mecenefm_change              -1.278833
    v2exl_legitideol_change         0.555000
    wdi_population_change      966727.000000
    v2x_polyarchy_change           -0.077500
    e_wbgi_gee_change              -0.248500
    e_wbgi_vae_change              -0.213667
    v2x_cspart_change              -0.180333
    dtype: float64

``` python
# Linear Regression with all variables in combined dataset:

# OLS model with all variables
m1_formula = 'v2x_corr ~ v2mecenefm + v2exl_legitideol + wdi_population + v2x_polyarchy + e_wbgi_gee + e_wbgi_vae + v2x_cspart'
model1_all_vars = smf.ols(formula=m1_formula, data=combined).fit()

print(model1_all_vars.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               v2x_corr   R-squared:                       0.754
    Model:                            OLS   Adj. R-squared:                  0.753
    Method:                 Least Squares   F-statistic:                     731.7
    Date:                Wed, 10 Dec 2025   Prob (F-statistic):               0.00
    Time:                        22:51:21   Log-Likelihood:                 793.51
    No. Observations:                1679   AIC:                            -1571.
    Df Residuals:                    1671   BIC:                            -1528.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept            0.5846      0.037     16.003      0.000       0.513       0.656
    v2mecenefm          -0.0066      0.006     -1.062      0.288      -0.019       0.006
    v2exl_legitideol    -0.0164      0.004     -4.216      0.000      -0.024      -0.009
    wdi_population    1.174e-10   2.38e-11      4.941      0.000    7.08e-11    1.64e-10
    v2x_polyarchy       -0.3046      0.054     -5.596      0.000      -0.411      -0.198
    e_wbgi_gee          -0.1925      0.007    -28.775      0.000      -0.206      -0.179
    e_wbgi_vae          -0.0320      0.016     -1.955      0.051      -0.064    9.84e-05
    v2x_cspart           0.0588      0.032      1.835      0.067      -0.004       0.122
    ==============================================================================
    Omnibus:                       21.037   Durbin-Watson:                   0.257
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               19.556
    Skew:                          -0.223   Prob(JB):                     5.67e-05
    Kurtosis:                       2.715   Cond. No.                     2.89e+09
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.89e+09. This might indicate that there are
    strong multicollinearity or other numerical problems.

``` python
model_coefficients = model1_all_vars.params

# Extract coefficients for independent variables (exclude the intercept
ind_var_coefficients = model_coefficients.drop('Intercept')

print(ind_var_coefficients)
```

    v2mecenefm         -6.634246e-03
    v2exl_legitideol   -1.637577e-02
    wdi_population      1.174399e-10
    v2x_polyarchy      -3.045729e-01
    e_wbgi_gee         -1.925307e-01
    e_wbgi_vae         -3.196652e-02
    v2x_cspart          5.882658e-02
    dtype: float64

``` python
contributions = {} # dictionary for contributions

# Loop to get each IV's coefficient
for variable, coefficient in ind_var_coefficients.items():
    change_var_name = f'{variable}_change'
    if change_var_name in avg_ca_changes: # check if change variable exists
        contributions[variable] = avg_ca_changes[change_var_name] * coefficient
    else:
        print(f"Unable to find a change for {variable} in avg_ca_changes.")

contributions_series = pd.Series(contributions) # pd transformation

print("Independent Variable contribution to v2x_corr change in Central America:")
print(contributions_series)
```

    Independent Variable contribution to v2x_corr change in Central America:
    v2mecenefm          0.008484
    v2exl_legitideol   -0.009089
    wdi_population      0.000114
    v2x_polyarchy       0.023604
    e_wbgi_gee          0.047844
    e_wbgi_vae          0.006830
    v2x_cspart         -0.010608
    dtype: float64

``` python
# Initial and final corruption average v2x_corr for C.A.
min_year_ca = ca_data['year'].min()
max_year_ca = ca_data['year'].max()

initial_v2x_corr_ca = ca_data[ca_data['year'] == min_year_ca]['v2x_corr'].mean()
final_v2x_corr_ca = ca_data[ca_data['year'] == max_year_ca]['v2x_corr'].mean()

# Waterfall chart legend
variable_labels = {
    'v2mecenefm': 'Mean Electoral Competitiveness Index',
    'v2exl_legitideol': 'Legitimacy of Electoral Process',
    'wdi_population': 'Population',
    'v2x_polyarchy': 'Electoral Democracy Index',
    'e_wbgi_gee': 'Government Effectiveness Index',
    'e_wbgi_vae': 'Voice and Accountability Index',
    'v2x_cspart': 'Civil Society Participation Index'
}

# Create a DataFrame for waterfall chart data
waterfall_data = pd.DataFrame({
    'measure': ['initial'] + list(contributions_series.index) + ['final'],
    'label': ['Initial Avg v2x_corr'] + [variable_labels.get(var, var) for var in contributions_series.index] + ['Final Avg v2x_corr'],
    'value': [initial_v2x_corr_ca] + list(contributions_series.values) + [final_v2x_corr_ca]
})

waterfall_data['type'] = 'relative' # changes
waterfall_data.loc[waterfall_data['measure'] == 'initial', 'type'] = 'absolute' # start
waterfall_data.loc[waterfall_data['measure'] == 'final', 'type'] = 'absolute' # end

display(waterfall_data)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | measure          | label                                | value     | type     |
|-----|------------------|--------------------------------------|-----------|----------|
| 0   | initial          | Initial Avg v2x_corr                 | 0.656667  | absolute |
| 1   | v2mecenefm       | Mean Electoral Competitiveness Index | 0.008484  | relative |
| 2   | v2exl_legitideol | Legitimacy of Electoral Process      | -0.009089 | relative |
| 3   | wdi_population   | Population                           | 0.000114  | relative |
| 4   | v2x_polyarchy    | Electoral Democracy Index            | 0.023604  | relative |
| 5   | e_wbgi_gee       | Government Effectiveness Index       | 0.047844  | relative |
| 6   | e_wbgi_vae       | Voice and Accountability Index       | 0.006830  | relative |
| 7   | v2x_cspart       | Civil Society Participation Index    | -0.010608 | relative |
| 8   | final            | Final Avg v2x_corr                   | 0.645667  | absolute |

</div>

``` python
fig = go.Figure(go.Waterfall(
    name = "Change in Corruption",
    orientation = "v",
    measure = waterfall_data['type'],
    x = waterfall_data['label'],
    textposition = "outside",
    text = [f'{val:.3f}' for val in waterfall_data['value']],
    y = waterfall_data['value'],
    connector = {"line": {"color": "rgb(63, 63, 63)"}}
))

fig.update_layout(
    title = "Average Corruption Index Change in Central America (2012-2022) (Waterfall Chart)",
    showlegend = True,
    height=600,
    width=1000,
    xaxis_title="Variable",
    yaxis_title="Corruption Index"
)

fig.show()
```

        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-2.35.2.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        

<div>                            <div id="65992257-74e4-45f5-ac3a-aa684631efe6" class="plotly-graph-div" style="height:600px; width:1000px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("65992257-74e4-45f5-ac3a-aa684631efe6")) {                    Plotly.newPlot(                        "65992257-74e4-45f5-ac3a-aa684631efe6",                        [{"connector":{"line":{"color":"rgb(63, 63, 63)"}},"measure":["absolute","relative","relative","relative","relative","relative","relative","relative","absolute"],"name":"Change in Corruption","orientation":"v","text":["0.657","0.008","-0.009","0.000","0.024","0.048","0.007","-0.011","0.646"],"textposition":"outside","x":["Initial Avg v2x_corr","Mean Electoral Competitiveness Index","Legitimacy of Electoral Process","Population","Electoral Democracy Index","Government Effectiveness Index","Voice and Accountability Index","Civil Society Participation Index","Final Avg v2x_corr"],"y":[0.6566666666666667,0.008484095042966149,-0.009088552605014266,0.00011353228137621177,0.02360439859139605,0.04784386973957708,0.006830179308851926,-0.010608393471943263,0.6456666666666666],"type":"waterfall"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"},"margin":{"b":0,"l":0,"r":0,"t":30}}},"title":{"text":"Average Corruption Index Change in Central America (2012-2022) (Waterfall Chart)"},"showlegend":true,"height":600,"width":1000,"xaxis":{"title":{"text":"Variable"}},"yaxis":{"title":{"text":"Corruption Index"}}},                        {"responsive": true}                    ).then(function(){
                            &#10;var gd = document.getElementById('65992257-74e4-45f5-ac3a-aa684631efe6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});
&#10;// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}
&#10;// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}
&#10;                        })                };                });            </script>        </div>

``` python
# 3. What is the relationship between electoral competitiveness and corruption levels across regions?
correlation = combined['v2mecenefm'].corr(combined['v2x_corr'], method='pearson')
print(f"Pearson Correlation between Mean Electoral Competitiveness Index and Corruption Index: {correlation:.4f}")
```

    Pearson Correlation between Mean Electoral Competitiveness Index and Corruption Index: -0.5929

``` python
plt.figure(figsize=(12, 8))
sns.regplot(data=combined, x='v2mecenefm', y='v2x_corr', scatter_kws={'alpha':0.6}, line_kws={'linestyle':'--', 'color':'red'})
plt.title('Relationship between Mean Electoral Competitiveness Index and Corruption Index')
plt.xlabel('Mean Electoral Competitiveness Index')
plt.ylabel('Corruption Index')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-17-output-1.png)

``` python
# Even though the previous plot show a clear negative correaltion between electroal competitiveness and corruption, I thought it would be interest to identify the regions where this relationship may be stronger or weaker. To do this, I added a color hue based on the regions defined in the V-Dem dataset. With this, I can see if certain regions deviate from the overall trend or if the relationship holds consistently across different parts of the world.

plt.figure(figsize=(14, 10))
sns.scatterplot(data=combined, x='v2mecenefm', y='v2x_corr', hue='region_class', alpha=0.6, s=50)
sns.regplot(data=combined, x='v2mecenefm', y='v2x_corr', scatter=False, color='red', line_kws={'linestyle':'--'})
plt.title('Relationship between Mean Electoral Competitiveness Index and Corruption Index by Region')
plt.xlabel('Mean Electoral Competitiveness Index')
plt.ylabel('Corruption Index')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-18-output-1.png)

``` python
# In the new regression plot, we can see how Europe leans toward high electoral competitiveness and low corruption, and while there are some European countries with higher corruption, the overall trend remains consistent across the region. On the other hand, Africa shows low electoral competitiveness and high corruption, with some exceptions.

# Interestingly enough, the Middle East is a region where some countries experience low electoral competitiveness but also relatively low corruption levels, deviating from the general trend observed in other regions. This suggests that factors other than electoral competitiveness may play a significant role in influencing corruption levels in this region.

# As for America, Asia, and Oceania, they generally follow the overall trend, but there are a few outliers in each region that deviate from the expected relationship between electoral competitiveness and corruption.
```

``` python
# Additional Analysis:
# America:
# # TS Plot for Sub-Regions within a Major Region
# Because a region and sub-region classification has already been created, this could easily be edited to plot any major region and its sub-regions
use_region = 'America' 
region_data = combined[combined['region_class'] == use_region].copy()

# Unique sub-regions within this major region for plotting
sub_regions_to_plot = region_data['sub_region_class'].unique()

time_series_sub = region_data.pivot_table(
    index='year',
    columns='sub_region_class',
    values='v2x_corr', # variable for analysis
    aggfunc='mean'
)

plt.figure(figsize=(12, 7))

for sub_region in sub_regions_to_plot:
    if sub_region in time_series_sub.columns:
        plt.plot(
            time_series_sub.index,
            time_series_sub[sub_region],
            marker='o',
            label=sub_region, # sub-region legend
            linewidth=2,
            alpha=0.8
        )

plt.title(f'Corruption Over Time - Sub-Regions within {use_region} (2012-2022)')
plt.xlabel('Year')
plt.ylabel('V-Dem Corruption Index (1 = Low Corruption)')
plt.legend(title=f'{use_region} Sub-Regions', loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-20-output-1.png)

``` python
# Europe
# Reusing code from previous cell
use_region = 'Europe' 
region_data = combined[combined['region_class'] == use_region].copy()

sub_regions_to_plot = region_data['sub_region_class'].unique()

time_series_sub = region_data.pivot_table(
    index='year',
    columns='sub_region_class',
    values='v2x_corr',
    aggfunc='mean'
)

plt.figure(figsize=(12, 7))

for sub_region in sub_regions_to_plot:
    if sub_region in time_series_sub.columns:
        plt.plot(
            time_series_sub.index,
            time_series_sub[sub_region],
            marker='o',
            label=sub_region,
            linewidth=2,
            alpha=0.8
        )

plt.title(f'Corruption Over Time - Sub-Regions within {use_region} (2012-2022)')
plt.xlabel('Year')
plt.ylabel('V-Dem Corruption Index (1 = Low Corruption)')
plt.legend(title=f'{use_region} Sub-Regions', loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-21-output-1.png)

``` python
# Moving forward and future work:
# For expanded work, it would be interesting to make the visualizations interactive and combine them with a dashboard to allow users to explore the data more deeply. Furthermore, the data could show different results based on the user's inputs and because everyone will look at different aspects of the data, this would enhance engagement and understanding.
```

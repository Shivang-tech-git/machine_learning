# print(df.columns)
# print(df.head())
# df.iloc[startrowindex:stoprowindex,startcolindex:stopcolindex]
# print(df.iloc[0])#returns the first row
# print(df.iloc[-1])#returns the last row
# print(df.iloc[1:5])#print row 1,2, 3 and 4
# print(df.iloc[8,21,53,76])#print row 8,21,53,76
# print(df.iloc[:,1:4])#return all rows for column 1,2,3
# print(df.iloc[:,[1,3,5,8]])#all rows, selective columns 1,3,5,8
# print(df.loc[df['first_name']=='France',['first_name','last_name','city']])
# df.loc[df['first_name']=='France','last_name']='andrew'
# print(df.loc[df['first_name']=='France',['first_name','last_name','city']])
# print(df.loc[df['email'].str.endswith('@gmail.com'),['first_name','email','web']])
# df1 = pd.read_excel('test2.xlsx',engine='openpyxl',sheet_name='Sheet2')
# df2 = pd.read_excel('test2.xlsx',engine='openpyxl',sheet_name='Sheet3')
# print(df1.head())
# print(df2.head())
#merge dataframes using common column
# df3 = pd.merge(df1,df2,on='department')
# df3 = pd.merge(df1,df2,on='department',how='left')
# df3 = pd.merge(df1,df2,on='department',how='right')
# df3 = pd.merge(df1,df2,on='department',how='outer')
#append data from one dataframe to another.
# df3 = pd.concat([df1,df2])
# import dateutil
# df = pd.read_csv('phone_data.csv')
#convert date column to date time format
# df['date'] = df['date'].apply(dateutil.parser.parse)
#get sum of duration column where item = call
# print(df.loc[df['item']=='call','duration'].sum())
# print(df['duration'].describe())
# print(df.head())
# print sum of duration for each unique month
# print(df.loc[df['item']=='call'].groupby(['month'])['duration'].sum())
# df.groupby(['month']).aggregate({'duration':[sum,max]})
# data = {'name':['tom','joe','charles','bob','steve','harry'],
#         'age':[20,21,20,21,21,20],
#         'mark':[10,81,67,58,69,70]
#         }
# df1 = pd.DataFrame(data)
# df1['grade'] = pd.cut(df1['mark'],bins=[0,50,60,75,90,100],labels=['E','D','C','B','A'])
import pandas as pd
import datetime
import xlwings
previous_month = datetime.datetime.today() - datetime.timedelta(days=30)
all_wb_path = {'giresh':{'path':'C:/Users/A134391/Desktop/july UK&I/Giresh.xlsb',
                         'Location':'India',
                         'Sub-Location':'Gurugram',
                         'Work Stream':'CAT Modelling'
                         },
               'veeksha':{'path':'C:/Users/A134391/Desktop/july UK&I/Veeksha.xlsb',
                          'Location':'India',
                          'Sub-Location':'Gurgaon and Bangalore',
                          'Work Stream':'International'
                      },
                'akshay':{'path':'C:/Users/A134391/Desktop/july UK&I/Akshay.xlsb',
                          'Location': 'India',
                          'Sub-Location': 'Gurgaon and Bangalore',
                          'Work Stream': 'Global Specialty'
                          },
                'gagan':{'path':'C:/Users/A134391/Desktop/july UK&I/Gagan.xlsb',
                         'nph': ['Time Consumed (Hrs)', 'SubTeam', 'Region', 'Category'],
                         'Location': 'India',
                         'Sub-Location': 'Gurgaon and Bangalore',
                         'Work Stream': 'International'
                         },
               'ops_poland':{'path':'C:/Users/A134391/Desktop/july UK&I/Ops Template Poland - July 2021.xlsb',
                            'Location': 'Poland',
                          'Sub-Location': 'Wroclaw',
                          'Work Stream': 'UK Operations'
                             },
               'global_energy':{'path':'C:/Users/A134391/Desktop/july UK&I/GLOBAL ENERGY I-Ops.xlsb',
                            'Location': 'Poland',
                          'Sub-Location': 'Wroclaw',
                          'Work Stream': 'UK Operations'
                                }
               }

all_ws_dict = {'individual':['Individual','A2'],
                'regional_individual':['Regional Individual','A2'],
               'production':['Production','A2'],
               'nph':['NPH','A2'],
               'key_highlights':['Key Highlights','A2'],
               'outstanding':['Outstanding','A2'],
               'challenges':['Challenges','A2'],
               'rewards':['Rewards','A2'],
               'external_error':['ExtrnErrUK','A2']}
# columns in worksheets
ind_cols = ['Region', 'Producing Country', 'Sub Team', 'Attendance (Days)', 'Non Production Time (Hrs)']
prod_cols = ['Region', 'Sub Team', 'Processing Time (Min)',
             'Delay(Y/N)', 'Exempted(Y/N)', 'QC Time (Min)', 'Volume', 'Producing Country']
nph_cols = ['Time Consumed (Hrs)', 'SubTeam', 'Region', 'Category']
# dataframes with desired columns
global_ind_df = pd.DataFrame(columns=['Country/stakeholder', 'Location',
                                      'Sub-Location', 'Work Stream', 'Sub Team',
                                      'Month', 'Year', 'Attendance (Days)',
                                      'Non Production Time (Hrs)'])
global_prod_df =pd.DataFrame(columns=['Country/Stakeholder', 'Location', 'Sub-Location', 'WorkStream', 'Sub Team',
                                      'Volume', 'Delayed Volume', 'Exempted Volume', 'Processing Time (In Hrs)',
                                      'QC Time (In Hrs)', 'Total Time', 'Month', 'Year'])
global_nph_df = pd.DataFrame(columns=['Country/Stakeholder','Location','Sub-Location','Work Stream','SubTeam',
                                      'Month','Year','NameOfActivity','Category','Time Consumed (Hrs)'])

region_filter = ['UK', 'UK&I','UK&I I-Ops','uk']

def add_col(df,Location,SubLocation,WorkStream):
    df.insert(0, 'Location', Location)
    df.insert(1, 'Sub-Location', SubLocation)
    df.insert(2, 'Work Stream', WorkStream)
    df.insert(3, 'Month', previous_month.strftime('%B'))
    df.insert(4, 'Year', previous_month.strftime('%Y'))

def ins_ops_template(global_ind_df, global_prod_df, global_nph_df):
    wb = xlwings.Book(new_col['path'])
    # ---------------------individual tab --------------------
    ind_ws = wb.sheets[all_ws_dict['regional_individual'][0]]
    ind_df = ind_ws.range(all_ws_dict['regional_individual'][1]).options(pd.DataFrame,header=1,index=False,
                                                                         expand='table').value
    if ind_df.shape[0] == 0:
        ind_ws = wb.sheets[all_ws_dict['individual'][0]]
        ind_df = ind_ws.range(all_ws_dict['individual'][1]).options(pd.DataFrame, header=1, index=False,
                                                                    expand='table').value
    # filter Region column for UK
    ind_df = ind_df[ind_df[ind_cols[0]].isin(region_filter)]
    # group by sub team, producing country and sum of attendance and NPH
    ind_df = ind_df[[ind_cols[2], ind_cols[1], ind_cols[3],
                   ind_cols[4]]].groupby([ind_cols[2], ind_cols[1]],
                    as_index=False).sum()
    # replace country name in producing country column
    ind_df = ind_df.replace({ind_cols[1]:{'United Kingdom':'UK'}})
    # rename producing country column
    ind_df.rename(columns={ind_cols[1]:'Country/stakeholder'}, inplace=True)
    # add columns Location, sub-Location, Work Stream, Month and Year
    add_col(ind_df,new_col['Location'],new_col['Sub-Location'],new_col['Work Stream'])
    # re-order the dataframe columns
    ind_df =ind_df[['Country/stakeholder','Location',
                                  'Sub-Location','Work Stream',ind_cols[2],
                                  'Month','Year',ind_cols[3],ind_cols[4]]]
    global_ind_df = global_ind_df.append(ind_df)

    # ---------------------  production tab --------------------
    # prod_cols = [0'Region', 1'Sub Team', 2'Processing Time (Min)',
    #              '3Delay(Y/N)', 4'Exempted(Y/N)', 5'QC Time (Min)', 6'Volume', 7'Producing Country']
    prod_ws = wb.sheets[all_ws_dict['production'][0]]
    prod_df = prod_ws.range(all_ws_dict['production'][1]).options(pd.DataFrame, header=1, index=False,
                                                                              expand='table').value
    prod_df = prod_df[prod_df[prod_cols[0]].isin(region_filter)]
    prod_df[prod_cols[2]] = pd.to_numeric(prod_df[prod_cols[2]], errors='coerce')
    prod_df[prod_cols[5]] = pd.to_numeric(prod_df[prod_cols[5]], errors='coerce')
    prod_df[prod_cols[2]] = (prod_df[prod_cols[2]].div(60))
    prod_df[prod_cols[5]] = (prod_df[prod_cols[5]].div(60))
    delay_df = prod_df.loc[prod_df[prod_cols[3]] == 'Y',
                                [prod_cols[1], prod_cols[7], prod_cols[6]]].groupby(
        [prod_cols[1], prod_cols[7]],as_index = False).sum()
    delay_df.rename(columns={prod_cols[6]:'Delayed Volume'},inplace=True)
    exempt_df = prod_df.loc[prod_df[prod_cols[4]] == 'Y',
                                [prod_cols[1], prod_cols[7], prod_cols[6]]].groupby(
        [prod_cols[1], prod_cols[7]],as_index = False).sum()
    exempt_df.rename(columns={prod_cols[6]:'Exempted Volume'},inplace=True)
    prod_df = prod_df[[prod_cols[1], prod_cols[7], prod_cols[6], prod_cols[2],
                                     prod_cols[5]]].groupby([prod_cols[1], prod_cols[7]],
                                                                         as_index=False).sum()
    prod_df = pd.merge(prod_df, delay_df,  how='left', left_on=[prod_cols[1], prod_cols[7]],
                              right_on = [prod_cols[1], prod_cols[7]])
    prod_df = pd.merge(prod_df, exempt_df, how='left', left_on=[prod_cols[1], prod_cols[7]],
                              right_on=[prod_cols[1], prod_cols[7]])
    prod_df['Total Time'] = prod_df[prod_cols[2]] + prod_df[prod_cols[5]]
    # replace country name in producing country column
    prod_df = prod_df.replace({prod_cols[7]: {'United Kingdom': 'UK'}})
    # rename columns
    prod_df.rename(columns={prod_cols[7]: 'Country/Stakeholder',prod_cols[2]: 'Processing Time (In Hrs)',
                            prod_cols[5]: 'QC Time (In Hrs)'}, inplace=True)
    # add columns Location, sub-Location, Work Stream, Month and Year
    add_col(prod_df, new_col['Location'], new_col['Sub-Location'], new_col['Work Stream'])
    # re-order the dataframe columns
    # global_prod_df = pd.DataFrame(columns=['Country/Stakeholder', 'Location', 'Sub-Location', 'WorkStream', 'Sub Team',
    #                                        'Volume', 'Delayed Volume', 'Exempted Volume', 'Processing Time (In Hrs)',
    #                                        'QC Time (In Hrs)', 'Total Time', 'Month', 'Year'])
    prod_df = prod_df[['Country/Stakeholder', 'Location', 'Sub-Location', 'Work Stream', 'Sub Team',
                                      'Volume', 'Delayed Volume', 'Exempted Volume', 'Processing Time (In Hrs)',
                                      'QC Time (In Hrs)', 'Total Time', 'Month', 'Year']]
    global_prod_df = global_prod_df.append(prod_df)
    # ---------------------  NPH tab --------------------
    # nph_cols = [0'Time Consumed (Hrs)', 1'SubTeam', 2'Region', 3'Category']
    nph_ws = wb.sheets[all_ws_dict['nph'][0]]
    nph_df = nph_ws.range(all_ws_dict['nph'][1]).options(pd.DataFrame,header=1,index=False,expand='table').value
    nph_df = nph_df[nph_df[nph_cols[2]].isin(region_filter)]
    nph_df[nph_cols[0]] = pd.to_numeric(nph_df[nph_cols[0]], errors='coerce')
    nph_df = nph_df[[nph_cols[1], nph_cols[3], nph_cols[0]]].groupby([nph_cols[1], nph_cols[3]],
                    as_index=False).sum()
    nph_df['Country/Stakeholder'] = 'UK'
    nph_df['NameOfActivity'] = ''
    # add columns Location, sub-Location, Work Stream, Month and Year
    add_col(nph_df, new_col['Location'], new_col['Sub-Location'], new_col['Work Stream'])
    # re-order the dataframe columns
    # global_nph_df = pd.DataFrame(columns=['Country/Stakeholder', 'Location', 'Sub-Location', 'Work stream', 'SubTeam',
    #                                       'Month', 'Year', 'NameOfActivity', 'Category', 'TimeConsumed'])
    nph_df = nph_df[['Country/Stakeholder', 'Location',
                     'Sub-Location', 'Work Stream', nph_cols[1],
                     'Month', 'Year', 'NameOfActivity', nph_cols[3], nph_cols[0]]]
    global_nph_df = global_nph_df.append(nph_df)
    return global_ind_df, global_prod_df, global_nph_df

def error_test(template):
    output_str = ''
    wb = xlwings.Book(new_col['path'])
    # check individual worksheet
    ind_ws = wb.sheets[all_ws_dict['regional_individual'][0]]
    ind_df = ind_ws.range(all_ws_dict['regional_individual'][1]).options(pd.DataFrame, header=1, index=False,
                                                                         expand='table').value
    if ind_df.shape[0] == 0:
        ind_ws = wb.sheets[all_ws_dict['individual'][0]]
        ind_df = ind_ws.range(all_ws_dict['individual'][1]).options(pd.DataFrame, header=1, index=False,
                                                                    expand='table').value
    col_list = [col for col in ind_df.columns]
    for name in ind_cols:
        if not name in col_list:
            output_str = '{0} \n {1} not found in row 2 of individual tab in {2}'.format(output_str, name, template)

    ws_array = ['production','nph']
    col_array = {0:prod_cols, 1:nph_cols}
    counter = 0
    # check all worksheets for header names in second row
    for sheet in ws_array:
        ws = wb.sheets[all_ws_dict[sheet][0]]
        df = ws.range(all_ws_dict[sheet][1]).options(pd.DataFrame, header=1, index=False,
                                                                             expand='table').value
        col_list = [col for col in df.columns]
        for name in col_array[counter]:
            if not name in col_list:
                output_str = '{0} \n {1} not found in row 2 of {2} tab in {3}'.format(output_str, name, sheet,template)
        counter += 1
    return output_str

if __name__ == '__main__':

    # for template in all_wb_path:
    #     print('Checking {}'.format(template))
    #     new_col = all_wb_path[template]
    #     output_str = error_test(template)
    #     print(output_str)

    for template in all_wb_path:
        new_col = all_wb_path[template]
        global_ind_df, global_prod_df, global_nph_df = ins_ops_template(global_ind_df, global_prod_df, global_nph_df)
    print('pause')


























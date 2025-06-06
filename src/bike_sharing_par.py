df = pd.read_csv('bike_sharing.csv')
y = df['count']
X = df.drop(columns=['count', 'datetime']) 



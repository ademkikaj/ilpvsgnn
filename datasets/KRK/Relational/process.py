import csv
import pandas as pd



# load data
df = pd.read_csv('datasets/KRK/Relational/krk.csv',delimiter=';')
print(df.head())

# split the dataframe into multiple dataframes by column names
df_class = df.loc[:,["id","class"]]
df_class.to_csv('datasets/KRK/Relational/class.csv',index=False)

df_whiteKing = df.loc[:,["id","white_king_file","white_king_rank"]]
df_whiteKing.rename(columns={"white_king_file":"file","white_king_rank":"rank"},inplace=True)
df_whiteKing.to_csv('datasets/KRK/Relational/white_king.csv',index=False)

df_whiteRook = df.loc[:,["id","white_rook_file","white_rook_rank"]]
df_whiteRook.rename(columns={"white_rook_file":"file","white_rook_rank":"rank"},inplace=True)
df_whiteRook.to_csv('datasets/KRK/Relational/white_rook.csv',index=False)

df_blackKing = df.loc[:,["id","black_king_file","black_king_rank"]]
df_blackKing.rename(columns={"black_king_file":"file","black_king_rank":"rank"},inplace=True)
df_blackKing.to_csv('datasets/KRK/Relational/black_king.csv',index=False)

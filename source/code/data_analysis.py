def missing_value(data): 
   print("Missing Value Analysis")
   print(data.isna().sum())
   if data.isna().sum().all() == 0:
    print("There are no missing values")
   else:
    print("There are some missing values")
   return 0 

def value_count(data): 
   print("Value Counts")
   label_counts = data["label"].value_counts()
   print(label_counts)
   return 0 


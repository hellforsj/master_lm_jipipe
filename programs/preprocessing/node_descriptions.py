import pandas as pd
import re

def prepare_dataset(path):
    df=pd.read_csv(path)
    cleaned_name=[]
    cleaned_description=[]
    for index, row in df.iterrows():
        try:
            if not (re.search("DEPRECATED",row["name"],re.IGNORECASE) or re.search("DEPRECATED",row["description"],re.IGNORECASE)):
                cleaned_name.append(row["name"])
                if re.search("The developer provided no description",row["description"]):
                    cleaned_description.append("")
                else:
                    cleaned_description.append(row["description"])
        except:
            cleaned_name.append(row["name"])
            cleaned_description.append("")
            pass
    return pd.DataFrame({"name":cleaned_name, "description": cleaned_description})
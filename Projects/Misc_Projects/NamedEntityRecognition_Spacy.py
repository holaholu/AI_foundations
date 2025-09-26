#pip install spacy
#python -m spacy download en_core_web_sm #this command is to download the pretrained english model
#import necessary libraries
import spacy
from spacy import displacy
import pandas as pd

#Load the english model
nlp = spacy.load("en_core_web_sm")

#Samples text for NER
text = """
Amazon announced its quarterly earnings on 2023-09-25.CEO Andy Jassy presented the earnings call and said the company is investing $4 billion in AI and cloud. Google, based in Mountain View, California, reported a loss of $1 billion in Q3 2023 in its financial report. The 2024 Summer Olympics will be held in Paris, France from July 26 to August 11, 2024.
"""

#Process the text
doc = nlp(text)

#function to extract entities
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'Entity': ent.text,
            'Label': ent.label_,
            'Explanation': spacy.explain(ent.label_)
        })
    return pd.DataFrame(entities)

#extract entities into a dataframe
entities_df = extract_entities(doc)
print('Extracted Named Entities: \n', entities_df)

#visualize the entities
displacy.render(doc, style='ent')

#save entities to a csv file
entities_df.to_csv('extracted_entities.csv', index=False)
print('Extracted entities saved to extracted_entities.csv')



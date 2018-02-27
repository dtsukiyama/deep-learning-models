## Phrase matching with spaCy

There are two ways to recognize entities: first, train your own named entity recognizer; second, simply match terms, spaCy can do both. When you don't have a significant amount of training data, matching is your best bet. This repo provides a wrapper around spaCy's phrase matching functionality.

## Set-up

Create a virtual environment or conda environment, install requirements.

```
virtualenv -p python3 env

source env/bin/activate

pip install -r requirements.txt

```

## Use

You will need a list of phrases you want to match on:

```python

mylist = ['Naruto','Konoha','Sasuke','Sakura','Kakashi','Orochimaru','Hokage','Land of Fire']

```

Import library:

```python
from recognizer import view, phraseRecognizer
```

You can view entity types with view:

```python

view()

{'CARDINAL': 'Numerals that do not fall under another type.',
 'DATE': 'Absolute or relative dates or periods.',
 'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
 'FACILITY': 'Buildings, airports, highways, bridges, etc.',
 'GPE': 'Countries, cities, states.',
 'LANGUAGE': 'Any named language.',
 'LAW': 'Named documents made into laws.',
 'LOC': 'Non-GPE locations, mountain ranges, bodies of water.',
 'MONEY': 'Monetary values, including unit.',
 'NORP': 'Nationalities or religious or political groups.',
 'ORDINAL': 'first, second, etc.',
 'ORG': 'Companies, agencies, institutions, etc.',
 'PERCENT': 'Percentage, including "%".',
 'PERSON': 'People, including fictional.',
 'PRODUCT': 'Objects, vehicles, foods, etc. (Not services.)',
 'QUANTITY': 'Measurements, as of weight or distance.',
 'TIME': 'Times smaller than a day.',
 'WORK_OF_ART': 'Titles of books, songs, etc.'}
```

phraseRecognizer has a couple of arguments:

Args: list of phrases to match, entity label, redesignate label with custom label
Returns: phrase matcher object
        
Use: labels attach an entity type to matched phrases. However these entities must adhere to spaCy's entity types.
You can designate your own label with mylabel, this simply creates a look-up table. 

```termMatcher``` ignore entity types; therefore this argument can be ignored. However ```spanMatcher``` will return entity type.
        
Simply match all phrases:

```python 
pr = phraseRecognizer(mylist)
```
             
Match all phrases or return phrases with entity type 'PERSON'  

```python 
pr = phraseRecognizer(naruto, label='PERSON')
```
        
Match all phrases and designate entity type 'PERSON' as 'ANIME'

```python 
pr = phraseRecognizer(naruto, label='PERSON',mylabel='ANIME')
```


Example:

```python
sample = 'Will Kakashi use his Sharingan?'

pr.termMatcher(sample)
'Kakashi'
```

```python
pr.spanMatcher(sample)
[('Kakashi', 'PERSON')]
```


```python

pr = phraseRecognizer(mylist, mylabel='ANIME')

sample = ["Where does Kakashi live in the Land of Fire?",
          "When is Naruto leaving?",
          "Sasuke is pretty annoying",
          "Did Orochimaru defeat the Hokage?"]


[pr.spanMatcher(b) for b in sample]

[[('Kakashi', 'ANIME'), ('Land of Fire', 'ANIME')],
 [('Naruto', 'ANIME')],
 [('Sasuke', 'ANIME')],
 [('Orochimaru', 'ANIME'), ('Hokage', 'ANIME')]]
 ```

## Flask server 

You can deploy the matcher as a service. This is already set to positive words, which come from this paper:

> Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing 
    and Comparing Opinions on the Web." Proceedings of the 14th 
    International World Wide Web conference (WWW-2005), May 10-14, 
    2005, Chiba, Japan.

```python
main.py
```

In another terminal you can run:

```
curl -i -H "Content-Type: application/json" -X POST -d '{"data":"This sushi is amazing, I approve. It is so awesome it deserves outsize distinction."}' http://0.0.0.0:8080/predict    
```

## Dockerize API

You can run this API in Docker.

```
docker build -t phraseMatcher .
```

Check images:

```
docker image ls
```

Run the app:

```
docker run -p 4000:8080 phraseMatcher 
```

You can get predictions by opening up another terminal:

```
curl -i -H "Content-Type: application/json" -X POST -d '{"data":"This sushi is amazing, I approve. It is so awesome it deserves outsize distinction."}' http://localhost:4000/predict
```

Handy command to remove all containers:

```
docker ps -a -q | xargs -n 1 -I {} sudo docker rm {}
```

Force remove an image:

```
docker rmi -f imageID
```


import pprint

from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token

nlp = English()

entities = {'PERSON':'People, including fictional.',
            'NORP':'Nationalities or religious or political groups.',
            'FACILITY':'Buildings, airports, highways, bridges, etc.',
            'ORG':'Companies, agencies, institutions, etc.',
            'GPE':'Countries, cities, states.',
            'LOC':'Non-GPE locations, mountain ranges, bodies of water.',
            'PRODUCT':'Objects, vehicles, foods, etc. (Not services.)',
            'EVENT':'Named hurricanes, battles, wars, sports events, etc.',
            'WORK_OF_ART':'Titles of books, songs, etc.',
            'LAW':'Named documents made into laws.',
            'LANGUAGE':'Any named language.',
            'DATE':'Absolute or relative dates or periods.',
            'TIME':'Times smaller than a day.',
            'PERCENT':'Percentage, including "%".',
            'MONEY':'Monetary values, including unit.',
            'QUANTITY':'Measurements, as of weight or distance.',
            'ORDINAL':'first, second, etc.',
            'CARDINAL':'Numerals that do not fall under another type.'}

def view():
    pprint.pprint(entities)
    
    
class phraseRecognizer(object):

    def __init__(self, phrases, label = 'PERSON', mylabel=None):
        """
        Args: list of phrases to match, entity label, redesignate label with custom label
        Returns: phrase matcher object
        
        Use: labels attach an entity type to matched phrases. However these entities must adhere to spaCy's entity types.
             You can designate your own label with mylabel, this simply creates a look-up table. 
             
        termMatcher ignore entity types; therefore this argument can be ignored. However spanMatcher will return entity type.
        
        # simply match all phrases
        pr = phraseRecognizer(naruto)
             
        # match all phrases or return phrases with entity type 'PERSON'     
        pr = phraseRecognizer(naruto, label='PERSON')
        
        # match all phrases and designate entity type 'PERSON' as 'ANIME'
        pr = phraseRecognizer(naruto, label='PERSON',mylabel='ANIME')
       
        """
        self.label = nlp.vocab.strings[label]
        self.entity = label
        patterns = [nlp(text) for text in phrases]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('myList', None, *patterns)   
        Token.set_extension('is_term', default=False)
        self.entity_table = {self.entity:mylabel}


    def termMatcher(self, text):
        """
        Args:  text we wish to match
        Returns: list of matched terms
        """
        terms = []
        doc = nlp(text)
        matches = self.matcher(doc)
        for ent_id, start, end in matches:
            terms.append(doc[start:end].text)
        return ', '.join([str(b) for b in terms])
    
    
    def spanMatcher(self, text):
        """
        From spaCy repository; attaches a label to entity. 
        """
        doc = nlp(text)
        matches = self.matcher(doc)
        spans = []  # keep the spans for later so we can merge them afterwards
        for _, start, end in matches:
            # Generate Span representing the entity & set label
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            # Set custom attribute on each token of the entity
            for token in entity:
                token._.set('is_term', True)
            # Overwrite doc.ents and add entity – be careful not to replace!
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
            span.merge()
            
        if not self.entity_table[self.entity]:
            return [(e.text, e.label_) for e in doc.ents]
        else:
            return [(e.text, self.entity_table[e.label_]) for e in doc.ents]  # don't forget to return the Doc!

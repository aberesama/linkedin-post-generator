import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from llm_helper import llm

def process_posts(raw_file_path, processed_file_path = ''):
    enriched_posts = []
    with open(raw_file_path, encoding = 'utf-8') as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_metadata(post['text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags[tag] for tag in current_tags}
        post['tags'] = list(new_tags) 

    with open(processed_file_path, encoding='utf-8', mode = "w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)

def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags. 
    1. Return a valid json. No preamble.
    2. Json object should have exctly three keys: line_count, language, and tags.
    3. Tags is an array of text tags. extract a maximum of two tags.
    4. The language should be in English.
    Here is the post on which you need to perform the extraction:
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input = {'post': post})

    res = {}
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(str(response.content))
    except OutputParserException as e:
        pass
    return res

def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    # Loop through each post and extract the tags
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  # Add the tags to the set

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of raw tags related to Artificial Intelligence, Machine Learning, and Data Science. You need to unify these tags based on the following requirements:

    1. Unify and Merge: Consolidate similar, synonymous, or sub-topic tags into a single, standardized parent category.
        Example 1: "Deep Learning", "DL", "Neural Nets", "ANN" should be merged into "Deep Learning"
        Example 2: "LLM", "Large Language Model", "GPT-4", "GenAI" should be merged into "Generative AI"
        Example 3: "Data Viz", "Plotting", "Dashboards", "Tableau" should be merged into "Data Visualization"
        Example 4: "Cleaning", "Wrangling", "Preprocessing" should be merged into "Data Engineering"

    2. Formatting: All unified tags must follow Title Case convention (e.g., "Computer Vision", "Natural Language Processing").
    3. Output Format: The output must be a valid JSON object only. Do not include markdown code blocks (```json), preambles, or explanations.
    Mapping: The JSON must map the specific original tag as the key to the unified tag as the value.
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(str(response.content))
        return res
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")

if __name__ == '__main__':
    process_posts('data/raw_posts.json', 'data/processed_posts.json')
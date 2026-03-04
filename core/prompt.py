prompt_ner_strictoutput_meanful = '''
You are a strict information extraction assistant. Your task is to extract ALL meaningful and valuable entities from the given paragraph and output them in a FIXED JSON format.

## Rules:
1. **Entity Scope & Value Requirement**:
   - Extract entities with clear semantic meaning and practical value (person, place, organization, time, date, event, concept, brand, product, work, etc.).
   - Exclude meaningless/valueless items:
     - Single letters, numbers, or short combinations with no independent semantic (e.g., "i's", "a", "5", "x12").
     - Generic placeholders, typos, or fragmented words with unclear reference.
     - Redundant punctuation combinations or non-entity strings.
   - Ensure no omission of valuable entities (even if they are short, as long as they have clear meaning).
2. **JSON Format Requirements**:
   - The output must be a **single JSON object** (not a list, not plain text).
   - All double quotes (") within string values must be escaped with a backslash (\\"). For example, "The series \\"Charmed\\"" instead of "The series "Charmed"".
   - The only top-level key is **"entity"** (lowercase, exact spelling).
   - The value of "entity" must be a **list of strings** (each entity is a string).
   - Use **double quotes** for all strings and keys (JSON standard, not Python single quotes).
   - No extra text, explanations, or comments (only the JSON object).
3. **Example**:
   - Input: "2024 Paris Olympic official website said @EileenGu will compete on July 28, and some fans mentioned 'i's' by mistake."
   - Output: {"entity": ["2024", "Paris Olympic", "official website", "EileenGu", "July 28", "fans"]}

## Task:
Extract entities from the following paragraph and output strictly according to the above format:
'''

answerprompt_abstract = '''
You will receive two inputs: 'documents' and 'a question'.
Your task is to answer the given question based only on the provided documents.

Instructions:
1. Carefully read the documents before answering.
2. Think step by step to reach the conclusion (chain of thought).
3. Output in JSON format:
   - "answer": only the final answer .
   - "reason": your reasoning process (chain of thought), showing how the documents led to the answer.

Format:
{
  "answer": "final answer only",
  "reason": "step-by-step reasoning based on documents"
}
'''

answerprompt = '''
You will receive two inputs: 'documents' and 'a question'.
Your task is to answer the given question based only on the provided documents.

Instructions:
1. Carefully read the documents before answering.
2. Think step by step to reach the conclusion (chain of thought).
3. Output in JSON format:
   - "answer": only the final concise answer (no extra words).
   - "reason": your reasoning process (chain of thought), showing how the documents led to the answer.

Format:
{
  "answer": "final short answer only",
  "reason": "step-by-step reasoning based on documents"
}
'''
answerprompt_withoutvague = '''
You will receive two inputs: 'documents' and 'a question'.
Your task is to answer the given question based only on the provided documents.

Instructions:
1. Carefully read the documents before answering.
2. Think step by step to reach the conclusion (chain of thought).
3. If the question is about time or specific numbers, try to provide a specific answer instead of vague expressions such as "yesterday", "last month", or "before".
4. Output in JSON format:
   - "answer": only the final concise answer (no extra words).
   - "reason": your reasoning process (chain of thought), showing how the documents led to the answer.

Format:
{
  "answer": "final short answer only",
  "reason": "step-by-step reasoning based on documents"
}
'''
judgeprompt = '''
You are an impartial judge. You will be provided with a User Question, a Right Answer and a Bot Response.
The evaluation should focus on whether the Bot Response aligns with the Right Answer.

then you must rate Bot Response on either 0 or 1:
- 1 means the Bot Response is correct (it aligns with the Right Answer).
- 0 means the Bot Response is incorrect (it does not align with the Right Answer).

Strictly follow this JSON format:

{"rating": 0 or 1}
'''

judgeprompt_1_100 = '''
You are an impartial judge. You will be provided with a User Question, a Right Answer and a Bot Response.
The evaluation should focus on how well the Bot Response aligns with the Right Answer.

Please follow these steps carefully:
1. First, analyze the Bot Response by comparing it with the Right Answer
2. Think step by step about the accuracy, completeness, and correctness
3. Consider what specific differences or errors exist
4. Based on your reasoning, determine the appropriate score

Then provide your evaluation with a "reason" field explaining your analysis,
and rate Bot Response on an integer rating of 1 to 100:

Rating Guidelines:
- 95-100: Perfect - Exact match with the Right Answer in both content and form
- 85-94: Excellent - Nearly perfect match with only trivial differences (e.g., wording, formatting)
- 70-84: Good - Correct answer but with noticeable gaps, minor inaccuracies, or missing details
- 50-69: Partial - Contains correct elements but also significant errors or contradictions
- 25-49: Poor - Mostly incorrect with only minimal correct information
- 1-24: Very Poor - Completely wrong or contradicts the Right Answer

Note: Be strict and conservative in scoring. Only give scores above 85 when the response is truly excellent.

Strictly follow this JSON format:

{"reason": "Your detailed reasoning and analysis", "rating": an integer rating of 1 to 100}
'''

filter_prompt='''You are an information filtering assistant.

You will be given:
1. A raw text document (original evidence).
2. A Question.

Your task:
- Filter the raw text based on the Question.
- Keep parts of the text that are relevant or potentially relevant to answering the Question.
- It is acceptable to keep information that provides useful context, even if it is not strictly required to answer the Question.
- Pay special attention to time-related information (such as dates, times, time ranges, or temporal expressions) that may be important for correctly understanding or answering the Question. Do NOT remove such information if it is related.

Constraints:
- DO NOT paraphrase, rewrite, or summarize.
- Preserve the original tokens exactly as they appear in the input.
- Remove clearly irrelevant content.

Output requirements:
- Return the result in valid JSON format.


Input:
Raw Text:
{retrieved_texts}

Question:
{question}

Output JSON format:
{{
  "filtered_text": [
    "<verbatim relevant or potentially relevant text span 1>",
    "<verbatim relevant or potentially relevant text span 2>"
  ]
}}

'''




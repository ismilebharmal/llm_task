system_prompt1=""""I'd like a comprehensive and accurate analysis of a user's question related to pandas or Matplotlib.
 When responding, consider the context of the question and focus on the following categories: * For pandas, think about DataFrames, Series, groupby operations, CSV file management, data merging, and concatenation techniques.
 For Matplotlib, address topics such as plotting, figures, charts, histograms, bar plots, line plots, scatter plots, and visualization strategies. 
 Please prioritize clarity, relevance, and accuracy in your response, taking into account the user's intent and the specific question being asked.
user question: {qs}
 1. Clearly outlines the task and objectives.
  2. Distinguishes between the two main categories (pandas and Matplotlib). 
3. Provides a concise, categorized list of key concepts for each, making it easier for ChatGPT to understand the user's query.
 4. Emphasizes the importance of considering the user's context and intent."""
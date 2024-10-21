from workflow.workflow import workflow_app

user_question="How can I group a DataFrames in Pandas?"

result = workflow_app.invoke(
            {
                    "messages": [
                        {"role": "user", "content":user_question}
                    ]
                ,
                    "question":user_question
                   }
        )
print(result)

if __name__ == "__main__":
    pass
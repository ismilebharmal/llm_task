from pydantic import BaseModel,Field
from typing import  Literal,Annotated
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    response:str
    finalResponse: str
    best_video:dict
    chunk_id:str
    error:str
    messages: Annotated[list[AnyMessage], add_messages]

# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

class FirstNodeResponse(BaseModel):
    """Submit the following objects to the user based on the query results."""
    datasource: Literal["pandas", "Matplotlib"] = Field(
        ...,
        description="Given a user question choose to route it relevant for pandas or matplotlib.",


)
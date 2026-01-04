from typing import List, Dict, Optional
from openai import OpenAI, AzureOpenAI
import os
import json
from retrieval_tool import RetrievalTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AgenticRAG:
    def __init__(self, retrieval_tool: RetrievalTool, model: str = None):
        self.retrieval_tool = retrieval_tool
        # if os.getenv("AZURE_OPENAI_API_KEY"):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        # else:
        #     self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #     self.model = model or os.getenv("MODEL", "gpt-4o-mini")

        self.conversation_history = []

    def reset_conversation(self):
        self.conversation_history = []

    def _create_system_prompt(self) -> str:
        return """You are a helpful AI assistant with access to a knowledge base. Your goal is to answer questions accurately based on retrieved information.

When answering:
1. Use the retrieve_knowledge tool to find relevant information from the knowledge base
2. Ground your answers in the retrieved documents - cite sources when possible
3. If the retrieved information doesn't fully answer the question, say so
4. Be honest about uncertainty - don't make up information
5. Provide clear, concise, and accurate answers

Always prioritize accuracy over completeness. It's better to say "I don't have enough information" than to hallucinate."""

    def _call_llm_with_tools(self, messages: List[Dict]) -> Dict:
        tools = [self.retrieval_tool.get_tool_definition()]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        return response

    def _execute_tool_calls(self, tool_calls) -> List[Dict]:
        tool_results = []

        for tool_call in tool_calls:
            if tool_call.function.name == "retrieve_knowledge":
                result = self.retrieval_tool.execute_tool_call(tool_call)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": result
                })

        return tool_results

    def _critic_reflection(self, question: str, answer: str, retrieved_docs: str) -> Dict:
        critic_prompt = f"""You are a critical evaluator. Evaluate the following answer for accuracy and grounding.

Question: {question}

Retrieved Documents:
{retrieved_docs}

Generated Answer:
{answer}

Evaluate the answer on:
1. **Accuracy**: Is the answer factually correct based on the documents?
2. **Grounding**: Is the answer well-supported by the retrieved documents?
3. **Completeness**: Does the answer fully address the question?
4. **Hallucination Risk**: Are there any claims not supported by the documents?

Provide your evaluation in JSON format:
{{
    "is_accurate": true/false,
    "is_grounded": true/false,
    "is_complete": true/false,
    "has_hallucination": true/false,
    "confidence_score": 0-10,
    "issues": ["list of any issues found"],
    "suggestions": ["list of suggestions for improvement"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                    "content": "You are a critical evaluator of AI-generated answers."},
                {"role": "user", "content": critic_prompt}
            ],
            response_format={"type": "json_object"}
        )

        try:
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
        except:
            return {
                "is_accurate": True,
                "is_grounded": True,
                "is_complete": True,
                "has_hallucination": False,
                "confidence_score": 7,
                "issues": [],
                "suggestions": []
            }

    def _improve_answer(self, question: str, original_answer: str, retrieved_docs: str, evaluation: Dict) -> str:
        improvement_prompt = f"""Based on the critic's feedback, improve the following answer.

Question: {question}

Retrieved Documents:
{retrieved_docs}

Original Answer:
{original_answer}

Critic Feedback:
- Issues: {', '.join(evaluation.get('issues', ['None']))}
- Suggestions: {', '.join(evaluation.get('suggestions', ['None']))}

Generate an improved answer that addresses the feedback while staying grounded in the retrieved documents."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": improvement_prompt}
            ]
        )

        return response.choices[0].message.content

    def answer_question(self, question: str, use_reflection: bool = True, verbose: bool = False) -> Dict:

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}\n")

        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": question}
        ]

        if verbose:
            print("Step 1: Calling LLM with retrieval tool...")

        response = self._call_llm_with_tools(messages)
        response_message = response.choices[0].message

        retrieved_docs = ""
        if response_message.tool_calls:
            if verbose:
                print(
                    f"Step 2: Executing {len(response_message.tool_calls)} tool call(s)...")

            messages.append(response_message)
            tool_results = self._execute_tool_calls(
                response_message.tool_calls)

            for result in tool_results:
                if result['name'] == 'retrieve_knowledge':
                    retrieved_docs = result['content']
                    if verbose:
                        print(
                            f"\nRetrieved Documents:\n{'-'*60}\n{retrieved_docs}\n{'-'*60}\n")

            messages.extend(tool_results)

            if verbose:
                print("Step 3: Generating answer from retrieved documents...")

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            answer = final_response.choices[0].message.content
        else:
            answer = response_message.content

        if verbose:
            print(f"\nInitial Answer:\n{answer}\n")

        evaluation = None
        final_answer = answer

        if use_reflection and retrieved_docs:
            if verbose:
                print("Step 4: Running self-reflection critic...")

            evaluation = self._critic_reflection(question, answer, retrieved_docs)

            if verbose:
                print(f"\nCritic Evaluation:")
                print(f"Confidence Score: {evaluation.get('confidence_score', 'N/A')}/10")
                print(f"Grounded: {evaluation.get('is_grounded', 'N/A')}")
                print(f"Accurate: {evaluation.get('is_accurate', 'N/A')}")
                if evaluation.get('issues'):
                    print(f"Issues: {', '.join(evaluation['issues'])}")

            if (evaluation.get('has_hallucination') or
                not evaluation.get('is_grounded') or
                    evaluation.get('confidence_score', 10) < 7):

                if verbose:
                    print("\nStep 5: Improving answer based on feedback...")

                final_answer = self._improve_answer(
                    question, answer, retrieved_docs, evaluation)

                if verbose:
                    print(f"\nImproved Answer:\n{final_answer}\n")

        return {
            "question": question,
            "answer": final_answer,
            "initial_answer": answer if use_reflection else None,
            "retrieved_docs": retrieved_docs,
            "evaluation": evaluation,
            "reflection_used": use_reflection and retrieved_docs != ""
        }

    def chat(self, message: str, verbose: bool = False) -> str:
        result = self.answer_question(
            message, use_reflection=True, verbose=verbose)
        return result['answer']

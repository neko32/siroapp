from sirochatora.sirochatora import Sirochatora, MessageBasedState
from sirochatora.rag.rag import LocalStorageRAG, RetrievalType
from sirochatora.util.siroutil import ConfJsonLoader
from os import environ
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Any
import operator
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, BaseOutputParser
from langgraph.graph import StateGraph, END


class Persona(BaseModel):
    name:str = Field(..., description = "ペルソナの名前")
    background:str = Field(..., description = "ペルソナの持つ背景")

class Personas(BaseModel):
    personas:list[Persona] = Field(
        default_factory = list, description = "ペルソナのリスト"
    )

class Interview(BaseModel):
    persona:Persona = Field(..., description = "インタビュー対称のペルソナ")
    question:str = Field(..., description = "インタビューでの質問")
    answer:str = Field(..., description = "インタビューでの回答")

class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory = list, description = "インタビュー結果のリスト"
    )

class EvaluationResult(BaseModel):
    evaluation_reason:str = Field(..., description = "判断の理由")
    is_information_sufficient:bool = Field(..., description = "情報が十分かどうか")

class InterviewerAgentState(BaseModel):
    user_req:str = Field(..., description = "ユーザーからのリクエスト")
    personas:Annotated[list[Persona], operator.add] = Field(
        default_factory = list, description = "生成されたペルソナのリスト"
    )
    interviews:Annotated[list[Interview], operator.add] = Field(
        default_factory = list, description = "実施されたインタビューのリスト"
    )
    requirements_doc:str = Field(default = "", description = "生成された要件定義")
    iteration:int = Field(default = 0, description = "ペルソナ生成とインタビューの反復回数")
    is_information_sufficient:bool = Field(default = False, description = "情報が十分かどうか")
    evaluation_reason:str = Field(default = "", description = "判定の理由")

class PersonaParser(BaseOutputParser[Personas]):

    def parse(self, input:str) -> Personas:
        ps = []
        input = input.replace("```json", "")
        input = input.replace("```", "")
        print(f"Got JSON STR as .. {input}")
        js = json.loads(input)
        names = js["name"]
        backgrounds = js["background"]
        for (name, background) in zip(names, backgrounds):
            ps.append(Persona(name = name, background = background))
        return Personas(personas = ps)

class EvaluationResultPerser(BaseOutputParser[EvaluationResult]):

    def parse(self, input:str) -> EvaluationResult:
        input = input.replace("```json", "")
        input = input.replace("```", "")
        print(f"Got JSON STR as .. {input}")
        js = json.loads(input)
        return EvaluationResult(
            evaluation_reason = js["reason"],
            is_information_sufficient = js["is_information_sufficient"]
        )
        
class PersonaGen:
    def __init__(self, sc:Sirochatora, k:int = 5):
        self._sc = sc
        self._k = k

    def run(self, user_req:str) -> Personas:
        parser = PersonaParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたはユーザーインタビュー用の多彩なペルソナを作成する専門家です"),
            ("human", f"""以下のユーザリクエストに関するインタビュー用に{self._k}人の多様なペルソナを生成してください。


            ユーザーリクエスト: {user_req}

            各ペルソナにはペルソナの持つ名前とペルソナの持つ背景を含めてください。年齢、性別、職業、技術的専門知識においては多様性を確保してください。
            """)
        ])
        chain = prompt | self._sc._llm | StrOutputParser() 
        chain_rez = chain.invoke({"user_req": user_req})

        ext_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは文章から指定されたデータを抽出してJSONに変換するタスクのエキスパートです"),
            ("human", """以下の文書から以下の属性を指定されたキーで抜き出してJSON形式で出力してください

            文書: {input_doc}

            JSONのデータは以下の通りでお願いします。

            name: 名前の値の配列
            background: 背景の値の配列

            例: 
            {{
                "name": ["最初の人の名前", "次の人の名前", ...],
                "background": ["最初の人の背景", "次の人の背景", ...]
            }} 
            """)
        ])
        ext_chain = ext_prompt | self._sc._llm | parser
        return ext_chain.invoke({"input_doc": chain_rez})
        #chain = prompt | self._sc._llm | parser
        #chain = prompt | self._sc._llm
        #return chain.invoke({"user_req": user_req}) # type: ignore

class Interviewer:
    def __init__(self, sc:Sirochatora):
        self._sc = sc

    def run(self, user_req:str, personas:list[Persona]) -> InterviewResult:
        questions = self.generate_question(user_req, personas)
        answers = self.generate_answers(personas = personas, questions = questions)
        interviews = self.create_interviews(personas = personas, questions = questions, answers = answers)
        return InterviewResult(interviews = interviews)

    def generate_question(self, user_req:str, personas:list[Persona]) -> list[str]:
        q_prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたはユーザ要件に基づいて適切な質問を生成する専門家です。"),
        ("human", """以下のペルソナに関連するユーザリクエストについて１つの質問を生成してください。

        ユーザリクエスト: {user_req}
        ペルソナ: {persona_name} - {persona_background}

        質問は具体的で、子のペルソナの視点から重要な情報を引き出すよう設計してください。    
        """)
        ])

        q_chain = q_prompt | self._sc._llm | StrOutputParser()
        qs = [
            {
                "user_req": user_req,
                "persona_name": persona.name,
                "persona_background": persona.background 
            } for persona in personas
        ]
        return q_chain.batch(qs)

    def generate_answers(self, personas:list[Persona], questions:list[str]) -> list[str]:
        ans_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたは以下のペルソナとして回答します。
            {persona_name} - {persona_background}
            """),
            ("human", "質問: {question}")
        ])
        ans_chain = ans_prompt | self._sc._llm | StrOutputParser()
        ans_q = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question
            } for persona, question in zip(personas, questions)
        ]
        return ans_chain.batch(ans_q)

    def create_interviews(self, personas:list[Persona], questions:list[str], answers:list[str]) -> list[Interview]:
        return [
            Interview(persona = persona, question = question, answer = answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]

class InformationEvaluator:
    def __init__(self, sc:Sirochatora):
        self._sc = sc

    def run(self, user_request:str, interviews:list[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。"),
            ("human", """以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まったかどうかを判断してください。

            ユーザーリクエスト: {user_request},
            インタビューの結果: {interview_result}
            """)
        ])
        chain = prompt | self._sc._llm | StrOutputParser()
        rez = {
                "user_request": user_request,
                "interview_result": "\n".join(
                    f"ペルソナ:{i.persona.name} - {i.persona.background}\n"
                    f"質問:{i.question}\n回答:{i.answer}\n"
                    for i in interviews
                )
        }

        ext_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは文章から指定されたデータを抽出してJSONに変換するタスクのエキスパートです"),
            ("human", """以下の文書から以下の属性を指定されたキーで抜き出してJSON形式で出力してください

            文書: {input_doc}

            JSONのデータは以下の通りでお願いします。

            is_information_sufficient: 最終的な要件文書を作るのに十分な情報が上記文書から得られたならTrue, 得られなかったらFalse
            reason: 上記がTrueかFalseになった理由を日本語で
            """)
        ])

        chain_rez = chain.invoke(rez)

        parser = EvaluationResultPerser()

        ext_chain = ext_prompt | self._sc._llm | parser
        return ext_chain.invoke({"input_doc": chain_rez})

        
class RequirementsDocumentGen:
    def __init__(self, sc:Sirochatora):
        self._sc = sc

    def run(self, user_request:str, interviews:list[Interview]) -> str:

        prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは収集した情報に基づいて要件文書を作成する専門家です。"),
            ("human", 
            """以下のユーザーリクエストと複数のペルソナからのインタビュー結果に基づいて要件文書を作成してください。

            ユーザーリクエスト: {user_request}
            インタビュー結果：{interview_results}
            なお、要件文書には以下のセクションを含めてください。

            1. プロジェクト概要
            2. 主要機能
            3. 非機能要件
            4. 制約条件
            5. ターゲットユーザー
            6. 優先順位
            7. リスクと軽減策
            出力は必ず日本語でお願いします。
            要件文書:
            """
            )
        ])

        req_doc_chain = prompt | self._sc._llm | StrOutputParser()

        return req_doc_chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                )
            }
        )

class Agent:
    def __init__(self, sc:Sirochatora, k:int = 5):
        self.persona_gen = PersonaGen(sc = sc, k = k)
        self.interviewer = Interviewer(sc)
        self.evaluator = InformationEvaluator(sc)
        self.req_gen = RequirementsDocumentGen(sc)

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewerAgentState)
        workflow.add_node("generate_personas", self.gf_generate_personas)
        workflow.add_node("conduct_interviews", self.gf_conduct_interviews)
        workflow.add_node("evaluate_information", self.gf_evaluate_info)
        workflow.add_node("generate_doc", self.gf_generate_doc)

        workflow.set_entry_point("generate_personas")

        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state:not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_doc"}
        )
        workflow.add_edge("generate_doc", END)
        return workflow.compile() # type: ignore

    def gf_generate_personas(self, state:InterviewerAgentState) -> 'dict[str, Any]':
        new_personas:Personas = self.persona_gen.run(state.user_req)
        print(new_personas)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1
        }

    def gf_conduct_interviews(self, state:InterviewerAgentState) -> 'dict[str, Any]':
        interviews:InterviewResult = self.interviewer.run(state.user_req, state.personas[-5:])
        return {"interviews": interviews.interviews}

    def gf_evaluate_info(self, state:InterviewerAgentState) -> 'dict[str, Any]':
        eval_result: EvaluationResult = self.evaluator.run(state.user_req, state.interviews)
        print(f"{eval_result.is_information_sufficient} - good to exit")
        return { 
            "is_information_sufficient": eval_result.is_information_sufficient, 
            "evaluation_reason": eval_result.evaluation_reason
        }

    def gf_generate_doc(self, state:InterviewerAgentState) -> 'dict[str, Any]':
        req_doc:str = self.req_gen.run(state.user_req, state.interviews)
        return {"requirements_doc": req_doc}

    def run(self, user_req:str) -> str:
        init = InterviewerAgentState(user_req = user_req)
        last = self.graph.invoke(init) # type: ignore
        return last["requirements_doc"]


def main():
    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]

    req = "健康管理のための携帯アプリを作りたい"

    #sc:Sirochatora = Sirochatora(role_def_conf = "study_role.json")
    #agent = Agent(sc)
    #print(agent.run(req))

    sc:Sirochatora = Sirochatora(role_def_conf = "study_role.json")
    sc.graph_init_simpletalk()
    print(sc.ask_with_graph("カモミールの効用について教えてください"))

if __name__ == "__main__":
    main()

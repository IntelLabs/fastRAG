from haystack import Document
from haystack.agents.base import AgentTokenStreamingHandler
from haystack.agents.conversational import ConversationalAgent
from haystack.lazy_imports import LazyImport
from haystack.nodes.retriever.multimodal.embedder import DOCUMENT_CONVERTERS

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image

from fastrag.prompters.invocation_layers.vqa_utils import base64_to_image


def _plan(self, query, current_step):
    # first resolve prompt template params
    template_params = self.prompt_parameters_resolver(
        query=query, agent=self, agent_step=current_step
    )

    # check for template parameters mismatch
    self.check_prompt_template(template_params)

    # invoke via prompt node
    prompt_node_response = self.prompt_node.prompt(
        prompt_template=self.prompt_template,
        stream_handler=AgentTokenStreamingHandler(self.callback_manager),
        **template_params,
        **self.response_generation_params,
        **self.image_data,
    )
    return prompt_node_response


# We override the standard ConversationalAgent._plan to allow for image input into the model
ConversationalAgent._plan = _plan


class ConversationRetrievalAugmenter:
    def __init__(self, agent, doc_pipeline, image_pipeline=None):
        self.agent = agent
        self.agent.response_generation_params = {}
        self.agent.image_data = {}

        self.doc_pipeline = doc_pipeline
        self.retrieval_prompt_template = """Here are documents that might relevant for our conversation:
{documents}
Read and use the documents in your response.
{query}
"""

        self.use_image_pipeline = image_pipeline is not None
        if self.use_image_pipeline:
            pillow_import.check()

            self.image_pipeline = image_pipeline

            def parse_image_doc(doc):
                try:
                    return base64_to_image(doc.content)
                except Exception as e:
                    return Image.open(doc.content)

            DOCUMENT_CONVERTERS["image"] = parse_image_doc

    def delete_all_data(self, params):
        """
        Delete all the data in the document stores.
        """
        doc_store = self.doc_pipeline.components["Retriever"].document_store
        doc_store.delete_all_documents()

        if self.use_image_pipeline:
            image_store = self.image_pipeline.components["retriever_text_to_image"].document_store
            image_store.delete_all_documents()

        del self.agent.image_data
        self.agent.image_data = {}

    def upload_docs(self, params):
        """
        Upload documents into the textual document store.
        """
        store = self.doc_pipeline.components["Retriever"].document_store
        docs = []
        for file_lines in params["file_texts"]:
            all_passages = []
            cur_passage = []
            for l in file_lines:
                line = l.replace("\r", "").replace("\n", "")
                if line == "":
                    if len(cur_passage) > 0:
                        cur_passage_text = " ".join(cur_passage)
                        all_passages.append(cur_passage_text)
                    cur_passage = []
                else:
                    cur_passage.append(line)

            if len(cur_passage) > 0:
                cur_passage_text = " ".join(cur_passage)
                all_passages.append(cur_passage_text)

            for passage in all_passages:
                doc = Document(content=passage)
                docs.append(doc)

        store.write_documents(docs)

    def upload_images(self, params):
        """
        Upload image into the image document store.
        """
        retriever = self.image_pipeline.components["retriever_text_to_image"]
        store = retriever.document_store

        # params['images'] is a list of base64 strings
        images = [
            Document(content=base64_str, content_type="image") for base64_str in params["images"]
        ]

        store.write_documents(images)
        store.update_embeddings(retriever=retriever)

    def load_chat(self, params):
        """
        Load chat history.
        """
        inputs = params["inputs"]
        outputs = params["outputs"]

        for inp, out in zip(inputs, outputs):
            self.agent.memory.save({"input": inp, "output": out})

    def clean_answer(self, ans):
        ans.answer = ans.answer.split("Human:")[0]
        return ans

    def modify_query_with_last_response(self, query):
        """
        Add the last response of the agent into the query
        """
        if len(self.agent.memory.list) > 0 and "AI" in self.agent.memory.list[-1]:
            ret_query = f"{self.agent.memory.list[-1]['AI']} {query}"
        else:
            ret_query = query

        return ret_query

    def add_context_to_user_request(self, query, params, ret_result):
        document_context = " ".join(
            [
                f"Document {len(params['inputs'])+1}-{i+1}:\n{x.content}\n\n"
                for i, x in enumerate(ret_result["documents"])
            ]
        )
        returned_docs = ret_result["documents"]
        user_prompt = self.retrieval_prompt_template.format(documents=document_context, query=query)
        return user_prompt, returned_docs

    def run(self, query, params, use_retrieval=False, use_image_retrieval=True):
        # clear user chat
        self.agent.memory.clear()

        # load user chat
        self.load_chat(params)

        user_prompt = query
        returned_docs = []

        # Extend the results with document retrieval
        if use_retrieval:
            # Modify the query with the last response from the agent
            ret_query = self.modify_query_with_last_response(query)

            # Ensure that the retriever parameters are present in the pipeline
            ret_pipe_params = {
                k: v for k, v in params["ret_pipe"].items() if k in self.doc_pipeline.components
            }

            ret_result = self.doc_pipeline.run(
                query=ret_query, params=ret_pipe_params, debug=params["debug"]
            )
            # add the documents to the user query
            user_prompt, returned_docs = self.add_context_to_user_request(query, params, ret_result)

        # Extend the results with image retrieval
        if use_image_retrieval:
            image_ret_result = self.image_pipeline.run(query=query)

            self.agent.image_data = {"images": [x.content for x in image_ret_result["documents"]]}

        self.agent.response_generation_params = params.get("generation_kwargs", {})

        agent_result = self.agent.run(query=user_prompt)

        # clean answers
        agent_result["answers"] = [self.clean_answer(ans) for ans in agent_result["answers"]]

        images_to_return = self.agent.image_data if use_image_retrieval else {}

        return agent_result, returned_docs, user_prompt, images_to_return

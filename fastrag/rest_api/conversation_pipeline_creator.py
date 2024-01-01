import os

import yaml
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptModel, PromptNode

import fastrag
from fastrag.rest_api.chat_template_initalizers import get_chat_prompt_template
from fastrag.rest_api.conversation_api import ConversationRetrievalAugmenter


def get_conversation_pipeline(args):
    model_api_key = os.getenv("HF_API_KEY", None)
    conversation_config = yaml.safe_load(open(args.config, "r"))

    use_image_pipeline = "image_pipeline_file" in conversation_config

    # Initialize chat model
    chat_model_config = conversation_config["chat_model"]
    if use_image_pipeline:
        # Ensure that the image chat model uses the correct implementation
        chat_model_config[
            "invocation_layer_class"
        ] = "fastrag.prompters.invocation_layers.vqa.VQAHFLocalInvocationLayer"

    PrompterModel = PromptModel(**chat_model_config)

    # Initialize a separate summary model, if specified

    if "summary_model" in conversation_config:
        SummaryPrompterModel = PromptModel(**conversation_config["summary_model"])
        SummaryPrompterModel.model_invocation_layer.pipe.tokenizer.pad_token = (
            SummaryPrompterModel.model_invocation_layer.pipe.tokenizer.eos_token
        )
        summary_prompt_node = PromptNode(SummaryPrompterModel, api_key=model_api_key)

        # Create memory summerizer
        summary_memory = ConversationSummaryMemory(
            summary_prompt_node, **conversation_config.get("summary_params", {})
        )
    else:
        summary_memory = None

    # Create PromptNodes
    prompt_node = PromptNode(PrompterModel, api_key=model_api_key)

    # Create the Conversational Agent
    conversational_agent = ConversationalAgent(prompt_node=prompt_node, memory=summary_memory)

    # Create the relevant chat template
    conversational_agent.prompt_template = get_chat_prompt_template(conversation_config)

    # load the document pipeline
    pipeline = (
        fastrag.load_pipeline(conversation_config["doc_pipeline_file"])
        if "doc_pipeline_file" in conversation_config
        else None
    )

    image_pipeline = (
        fastrag.load_pipeline(conversation_config["image_pipeline_file"])
        if use_image_pipeline
        else None
    )

    conv_ret_agent = ConversationRetrievalAugmenter(
        agent=conversational_agent, doc_pipeline=pipeline, image_pipeline=image_pipeline
    )

    return conv_ret_agent

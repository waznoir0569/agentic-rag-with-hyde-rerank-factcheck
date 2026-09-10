from uuid import UUID

from app.auth.dependencies import CurrentUserDep
from app.db.main import SessionDep
from app.threads import service as thread_service
from fastapi import APIRouter

from . import service as chat_service
from .schemas import ChatStreamResponse, Message, PersonaQuestionnaire, PromptInput

chat_router = APIRouter()


@chat_router.post("/")
async def simple_chat_stream(prompt_input: PromptInput):
    return ChatStreamResponse(await chat_service.simple_chat_stream(prompt_input))


@chat_router.post("/persona")
async def create_persona(questionnaire: PersonaQuestionnaire):
    from .persona import assign_cluster, behavioral_scoring, build_persona_system_prompt

    scores = behavioral_scoring(**questionnaire.model_dump())
    cluster_result = assign_cluster(scores)

    return {
        "scores": {
            "gamma_short": scores[0],
            "alpha_short": scores[1],
            "delta_short": scores[3],
        },
        "cluster_id": cluster_result["cluster_id"],
        "cluster_name": cluster_result["cluster_name"],
        "persona_prompt": build_persona_system_prompt(cluster_result),
    }


@chat_router.post("/{thread_id}")
async def chat_stream(thread_id: UUID, prompt_input: PromptInput, current_user: CurrentUserDep, session: SessionDep):
    await thread_service.get_thread(thread_id, current_user.id, session)
    return ChatStreamResponse(
        await chat_service.chat_stream(thread_id, prompt_input, current_user.id),
    )


@chat_router.get("/{thread_id}", response_model=list[Message])
async def get_chat_history(thread_id: UUID, current_user: CurrentUserDep, session: SessionDep):
    await thread_service.get_thread(thread_id, current_user.id, session)
    return await chat_service.get_chat_history(thread_id, current_user.id)

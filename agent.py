from openai.types.chat import ChatCompletionMessageParam


class Agent:
    """A lightweight LLM agent description."""

    def __init__(self, title: str, expertise: str, goal: str, role: str, model: str) -> None:
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model

    @property
    def prompt(self) -> str:
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
        )

    @property
    def message(self) -> ChatCompletionMessageParam:
        return {
            "role": "system",
            "content": self.prompt,
        }

    def __hash__(self) -> int:
        return hash((self.title, self.expertise, self.goal, self.role, self.model))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return False
        return (
            self.title == other.title
            and self.expertise == other.expertise
            and self.goal == other.goal
            and self.role == other.role
            and self.model == other.model
        )

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return self.title
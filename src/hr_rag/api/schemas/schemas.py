
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    # bcrypt only uses the first 72 bytes anyway; 8 is a sane floor
    password: str = Field(..., min_length=8, max_length=72)
    full_name: str = Field(..., min_length=1, max_length=200)


class ProvisionUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=72)
    full_name: str = Field(..., min_length=1, max_length=200)
    role: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    category: str | None = None


class SourceChunk(BaseModel):
    category: str
    policy_doc_id: str
    title: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    cached: bool
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    pipeline_loaded: bool

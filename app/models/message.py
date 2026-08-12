"""Model for a single raw message as read from the input CSV."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RawMessage(BaseModel):
    """A single validated message.

    Attributes:
        message_id: Unique identifier, preserved exactly as found in the CSV.
        timestamp: Parsed message timestamp.
        sender: Sender of the message.
        message: The message content.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    timestamp: datetime
    sender: str
    message: str

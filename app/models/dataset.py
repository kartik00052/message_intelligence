"""Models describing the validated input dataset as a whole."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DatasetStatistics(BaseModel):
    """Summary statistics for a validated message dataset.

    Attributes:
        total_messages: Number of messages in the dataset.
        unique_message_ids: Number of distinct message IDs.
        earliest_timestamp: Earliest message timestamp.
        latest_timestamp: Latest message timestamp.
        empty_message_count: Number of messages with empty content.
        empty_sender_count: Number of messages with an empty sender.
    """

    total_messages: int
    unique_message_ids: int
    earliest_timestamp: datetime
    latest_timestamp: datetime
    empty_message_count: int
    empty_sender_count: int

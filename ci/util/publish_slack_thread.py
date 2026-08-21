#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import sys
import time
from http import client
from urllib import error, request


class PublishError(RuntimeError):
    pass


SLACK_TIMESTAMP = re.compile(r"[0-9]+\.[0-9]+")


class NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise error.HTTPError(
            req.full_url,
            code,
            "redirects are not allowed",
            headers,
            fp,
        )


def load_thread(stream):
    try:
        thread = json.load(stream)
    except (UnicodeError, json.JSONDecodeError) as exception:
        raise PublishError("thread input is not valid JSON") from exception

    if not isinstance(thread, dict):
        raise PublishError("thread input must be an object")
    overview = thread.get("overview")
    replies = thread.get("replies")
    if not isinstance(overview, str) or not overview:
        raise PublishError("thread overview must be a non-empty string")
    if (
        not isinstance(replies, list)
        or not replies
        or any(not isinstance(reply, str) or not reply for reply in replies)
    ):
        raise PublishError("thread replies must be a non-empty array of strings")
    return overview, replies


def post_message(token, channel_id, text, thread_ts=None):
    payload = {
        "channel": channel_id,
        "text": text,
        "parse": "none",
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if thread_ts is not None:
        payload["thread_ts"] = thread_ts

    api_request = request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    opener = request.build_opener(NoRedirectHandler())

    for attempt in range(3):
        try:
            with opener.open(api_request, timeout=30) as response:
                result = json.load(response)
            break
        except error.HTTPError as exception:
            status = exception.code
            retry_after_header = exception.headers.get("Retry-After", "1")
            exception.close()
            if status != 429 or attempt == 2:
                raise PublishError(f"Slack returned HTTP {status}") from exception
            try:
                retry_after = max(1, int(retry_after_header))
            except ValueError:
                retry_after = 1
            time.sleep(retry_after)
        except (error.URLError, OSError, client.HTTPException) as exception:
            raise PublishError(
                "Slack request failed before receiving a complete response"
            ) from exception
        except (UnicodeError, json.JSONDecodeError) as exception:
            raise PublishError("Slack returned an invalid JSON response") from exception

    if not isinstance(result, dict):
        raise PublishError("Slack returned an invalid JSON response")
    if result.get("ok") is not True:
        slack_error = result.get("error", "unknown_error")
        raise PublishError(f"Slack API error: {slack_error}")
    message_ts = result.get("ts")
    if not isinstance(message_ts, str) or not message_ts:
        raise PublishError("Slack response did not contain a message timestamp")
    return message_ts


def main():
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    thread_ts = os.environ.get("SLACK_THREAD_TS")
    if not token:
        raise PublishError("SLACK_BOT_TOKEN is required")
    if not channel_id:
        raise PublishError("SLACK_CHANNEL_ID is required")
    if thread_ts and SLACK_TIMESTAMP.fullmatch(thread_ts) is None:
        raise PublishError("SLACK_THREAD_TS is not a valid Slack timestamp")

    overview, replies = load_thread(sys.stdin)
    if thread_ts:
        parent_ts = thread_ts
        replies.insert(0, overview)
    else:
        try:
            parent_ts = post_message(token, channel_id, overview)
        except PublishError as exception:
            raise PublishError(f"parent message failed: {exception}") from exception

    reply_count = len(replies)
    for index, reply in enumerate(replies, start=1):
        time.sleep(1)
        try:
            post_message(token, channel_id, reply, parent_ts)
        except PublishError as exception:
            raise PublishError(
                f"reply {index} of {reply_count} failed: {exception}"
            ) from exception


if __name__ == "__main__":
    try:
        main()
    except PublishError as exception:
        print(f"error: {exception}", file=sys.stderr)
        raise SystemExit(1)

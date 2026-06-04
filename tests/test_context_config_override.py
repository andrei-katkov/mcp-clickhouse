"""Tests for context state-based ClickHouse client configuration overrides."""

import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.server.dependencies import get_context
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from mcp_clickhouse.mcp_server import (
    CLIENT_CONFIG_OVERRIDES_KEY,
    create_clickhouse_client,
    mcp,
    run_query,
)


load_dotenv()

requires_clickhouse = pytest.mark.skipif(
    not all(
        [
            os.getenv("CLICKHOUSE_HOST"),
            os.getenv("CLICKHOUSE_USER"),
            os.getenv("CLICKHOUSE_PASSWORD"),
        ]
    ),
    reason="ClickHouse environment variables not set",
)


def _base_client_config() -> dict:
    return {
        "host": "clickhouse.example.test",
        "port": 8443,
        "username": "huginn",
        "password": "secret",
        "interface": "https",
        "secure": True,
        "verify": True,
        "connect_timeout": 30,
        "send_receive_timeout": 300,
        "client_name": "mcp_clickhouse",
    }


def _mock_clickhouse_config():
    config = MagicMock()
    config.get_client_config.return_value = _base_client_config()
    config.allow_write_access = False
    config.allow_drop = False
    return config


def _mock_clickhouse_query_client():
    client = MagicMock(server_version="24.1", server_settings={})
    result = MagicMock()
    result.column_names = ["one"]
    result.result_rows = [[1]]
    client.query.return_value = result
    return client


class ConfigOverrideMiddleware(Middleware):
    """Test middleware that sets ClickHouse client config overrides."""

    def __init__(self, overrides: dict):
        self.overrides = overrides

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        """Set config overrides in context state before tool execution."""
        ctx = get_context()
        ctx.set_state(CLIENT_CONFIG_OVERRIDES_KEY, self.overrides)
        return await call_next(context)


class TestConfigOverrideUnit:
    """Unit tests for the config override merge logic in create_clickhouse_client."""

    def test_overrides_merged_into_client_config(self):
        """Verify overrides from context state are merged into the client config."""
        mock_ctx = MagicMock()
        mock_ctx.get_state.return_value = {"connect_timeout": 99, "send_receive_timeout": 199}

        with (
            patch("mcp_clickhouse.mcp_server.get_config", return_value=_mock_clickhouse_config()),
            patch("mcp_clickhouse.mcp_server.clickhouse_connect") as mock_cc,
            patch("mcp_clickhouse.mcp_server.get_context", return_value=mock_ctx),
        ):
            mock_cc.get_client.return_value = _mock_clickhouse_query_client()
            create_clickhouse_client()

        call_kwargs = mock_cc.get_client.call_args[1]
        assert call_kwargs["connect_timeout"] == 99
        assert call_kwargs["send_receive_timeout"] == 199

    def test_empty_overrides_no_change(self):
        """Empty overrides dict should not alter the base config."""
        mock_ctx = MagicMock()
        mock_ctx.get_state.return_value = {}

        with (
            patch("mcp_clickhouse.mcp_server.get_config", return_value=_mock_clickhouse_config()),
            patch("mcp_clickhouse.mcp_server.clickhouse_connect") as mock_cc,
            patch("mcp_clickhouse.mcp_server.get_context", return_value=mock_ctx),
        ):
            mock_cc.get_client.return_value = _mock_clickhouse_query_client()
            create_clickhouse_client()

        call_kwargs = mock_cc.get_client.call_args[1]
        assert "host" in call_kwargs
        assert "username" in call_kwargs

    def test_no_overrides_in_context(self):
        """When context state has no overrides, base config is used as-is."""
        mock_ctx = MagicMock()
        mock_ctx.get_state.return_value = None

        with (
            patch("mcp_clickhouse.mcp_server.get_config", return_value=_mock_clickhouse_config()),
            patch("mcp_clickhouse.mcp_server.clickhouse_connect") as mock_cc,
            patch("mcp_clickhouse.mcp_server.get_context", return_value=mock_ctx),
        ):
            mock_cc.get_client.return_value = _mock_clickhouse_query_client()
            create_clickhouse_client()

        call_kwargs = mock_cc.get_client.call_args[1]
        assert "host" in call_kwargs

    def test_run_query_applies_role_override_across_executor_thread(self):
        """run_query must preserve request overrides when work moves to QUERY_EXECUTOR."""
        mock_ctx = MagicMock()
        mock_ctx.get_state.return_value = {
            "settings": {
                "role": "merchant_role_123",
            }
        }
        main_thread = threading.current_thread()

        def get_context_only_in_request_thread():
            if threading.current_thread() is main_thread:
                return mock_ctx
            raise RuntimeError("No FastMCP context in executor thread")

        with (
            patch("mcp_clickhouse.mcp_server.get_config", return_value=_mock_clickhouse_config()),
            patch(
                "mcp_clickhouse.mcp_server.get_mcp_config",
                return_value=SimpleNamespace(query_timeout=5),
            ),
            patch("mcp_clickhouse.mcp_server.clickhouse_connect") as mock_cc,
            patch(
                "mcp_clickhouse.mcp_server.get_context",
                side_effect=get_context_only_in_request_thread,
            ),
        ):
            mock_cc.get_client.return_value = _mock_clickhouse_query_client()
            run_query("SELECT merchant FROM analytics.bundle_product_sales")

        call_kwargs = mock_cc.get_client.call_args.kwargs
        assert call_kwargs["settings"] == {"role": "merchant_role_123"}

    def test_invalid_override_type_ignored_with_warning(self, caplog):
        """Invalid override values should not reach clickhouse_connect."""
        mock_ctx = MagicMock()
        mock_ctx.get_state.return_value = ["not", "a", "dict"]

        with (
            patch("mcp_clickhouse.mcp_server.get_config", return_value=_mock_clickhouse_config()),
            patch("mcp_clickhouse.mcp_server.clickhouse_connect") as mock_cc,
            patch("mcp_clickhouse.mcp_server.get_context", return_value=mock_ctx),
            caplog.at_level("WARNING", logger="mcp-clickhouse"),
        ):
            mock_cc.get_client.return_value = _mock_clickhouse_query_client()
            create_clickhouse_client()

        call_kwargs = mock_cc.get_client.call_args.kwargs
        assert "settings" not in call_kwargs
        assert (
            f"{CLIENT_CONFIG_OVERRIDES_KEY} must be a dict, got list. Ignoring."
            in caplog.text
        )

    def test_no_request_context_falls_back_to_defaults(self):
        """Outside a request context, base config is used."""
        with (
            patch("mcp_clickhouse.mcp_server.get_config", return_value=_mock_clickhouse_config()),
            patch("mcp_clickhouse.mcp_server.clickhouse_connect") as mock_cc,
        ):
            mock_cc.get_client.return_value = _mock_clickhouse_query_client()
            create_clickhouse_client()

        call_kwargs = mock_cc.get_client.call_args[1]
        assert "host" in call_kwargs
        assert "settings" not in call_kwargs


@pytest.fixture
def mcp_server():
    """Return the MCP server instance for testing."""
    return mcp


@requires_clickhouse
class TestConfigOverrideIntegration:
    """Integration tests that verify overrides work end-to-end with ClickHouse."""

    @pytest.mark.asyncio
    async def test_context_state_config_override(self, mcp_server):
        test_overrides = {
            "connect_timeout": 99,
            "send_receive_timeout": 199,
        }
        middleware = ConfigOverrideMiddleware(test_overrides)
        mcp_server.add_middleware(middleware)

        try:
            async with Client(mcp_server) as client:
                result = await client.call_tool("list_databases", {})
                assert len(result.content) >= 1
        finally:
            if middleware in mcp_server.middleware:
                mcp_server.middleware.remove(middleware)

    @pytest.mark.asyncio
    async def test_context_state_empty_override(self, mcp_server):
        middleware = ConfigOverrideMiddleware({})
        mcp_server.add_middleware(middleware)

        try:
            async with Client(mcp_server) as client:
                result = await client.call_tool("list_databases", {})
                assert len(result.content) >= 1
        finally:
            if middleware in mcp_server.middleware:
                mcp_server.middleware.remove(middleware)

    @pytest.mark.asyncio
    async def test_context_state_no_override(self, mcp_server):
        async with Client(mcp_server) as client:
            result = await client.call_tool("list_databases", {})
            assert len(result.content) >= 1

    @pytest.mark.asyncio
    async def test_multiple_tools_with_override(self, mcp_server):
        middleware = ConfigOverrideMiddleware({"connect_timeout": 88})
        mcp_server.add_middleware(middleware)

        try:
            async with Client(mcp_server) as client:
                result1 = await client.call_tool("list_databases", {})
                assert len(result1.content) >= 1

                result2 = await client.call_tool("list_databases", {})
                assert len(result2.content) >= 1
        finally:
            if middleware in mcp_server.middleware:
                mcp_server.middleware.remove(middleware)

    @pytest.mark.asyncio
    async def test_override_inherits_from_env(self, mcp_server):
        middleware = ConfigOverrideMiddleware({"connect_timeout": 77})
        mcp_server.add_middleware(middleware)

        try:
            async with Client(mcp_server) as client:
                result = await client.call_tool("list_databases", {})
                assert len(result.content) >= 1
        finally:
            if middleware in mcp_server.middleware:
                mcp_server.middleware.remove(middleware)

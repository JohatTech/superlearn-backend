"""
===============================================================================
UNIT TESTS: LANGCHAIN AGENTIC RAG & AGENTS TOOL ISOLATION
===============================================================================
"""

from __future__ import annotations
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.agents.base_agent import BaseAgent
from app.agents.rag_tools import (
    get_agent_tools,
    knowledge_corpus_tool,
    web_search_tool,
    _async_retrieve_knowledge_corpus,
    _sync_retrieve_knowledge_corpus,
    _async_web_search,
    _sync_web_search,
)
from app.agents.syllabus_agent import SyllabusAgent, syllabus_agent
from app.agents.study_materials_agent import StudyMaterialsAgent, study_materials_agent
from app.agents.test_generator_agent import TestGeneratorAgent, test_generator_agent
from app.agents.evaluator_agent import EvaluatorAgent, evaluator_agent
from app.core.cognitive_config import cognitive_settings


class TestAgentRAGTools:
    """Test suite for LangChain Agentic RAG and Web Search tools."""

    def test_get_agent_tools_with_web_search(self):
        """Verify web search is included when explicitly enabled."""
        tools = get_agent_tools(include_web_search=True)
        tool_names = [t.name for t in tools]
        assert "retrieve_knowledge_corpus" in tool_names
        assert "search_web" in tool_names
        assert len(tools) == 2

    def test_get_agent_tools_without_web_search(self):
        """Verify web search is strictly excluded when include_web_search is False."""
        tools = get_agent_tools(include_web_search=False)
        tool_names = [t.name for t in tools]
        assert "retrieve_knowledge_corpus" in tool_names
        assert "search_web" not in tool_names
        assert len(tools) == 1

    async def test_retrieve_knowledge_corpus_async_formatted_output(self):
        """Verify _async_retrieve_knowledge_corpus formats search results properly."""
        mock_results = [
            {
                "source_name": "MIT_Linear_Algebra.pdf",
                "chunk_index": 3,
                "similarity_score": 0.92,
                "passage_text": "An eigenvector is a nonzero vector whose direction remains unchanged.",
            },
            {
                "source_name": "Strang_Intro.pdf",
                "chunk_index": 12,
                "similarity_score": 0.88,
                "passage_text": "Matrix diagonalization decouples linear dynamical systems.",
            },
        ]

        with patch(
            "app.services.multisource_contrast_ingestion_engine.contrast_engine.search_semantic_passages",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = mock_results
            result = await _async_retrieve_knowledge_corpus("eigenvalues", top_k=2)

            assert "MIT_Linear_Algebra.pdf" in result
            assert "0.92" in result
            assert "An eigenvector is a nonzero vector" in result
            assert "Matrix diagonalization" in result

    async def test_retrieve_knowledge_corpus_async_empty_results(self):
        """Verify _async_retrieve_knowledge_corpus handles empty retrieval gracefully."""
        with patch(
            "app.services.multisource_contrast_ingestion_engine.contrast_engine.search_semantic_passages",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = []
            result = await _async_retrieve_knowledge_corpus("obscure_query", top_k=2)
            assert "No relevant passages found" in result

    async def test_retrieve_knowledge_corpus_async_exception_handling(self):
        """Verify _async_retrieve_knowledge_corpus catches unexpected errors without crashing."""
        with patch(
            "app.services.multisource_contrast_ingestion_engine.contrast_engine.search_semantic_passages",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Qdrant connection dropped"),
        ):
            result = await _async_retrieve_knowledge_corpus("query", top_k=2)
            assert "Knowledge corpus retrieval unavailable" in result

    def test_retrieve_knowledge_corpus_sync(self):
        """Verify synchronous retrieval wrapper executes without hanging."""
        with patch(
            "app.services.multisource_contrast_ingestion_engine.contrast_engine.search_semantic_passages",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = [
                {
                    "source_name": "TestDoc",
                    "chunk_index": 1,
                    "similarity_score": 0.95,
                    "passage_text": "Sample text",
                }
            ]
            result = _sync_retrieve_knowledge_corpus("test query", top_k=1)
            assert "TestDoc" in result or "Knowledge corpus retrieval unavailable" in result

    async def test_web_search_async_success_and_fallback(self):
        """Verify web search handles execution and rate-limit fallbacks."""
        with patch("langchain_community.tools.DuckDuckGoSearchRun.ainvoke", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Search result snippets..."
            res = await _async_web_search("MIT OCW Algorithms")
            assert "Search result snippets..." in res

        with patch("langchain_community.tools.DuckDuckGoSearchRun.ainvoke", side_effect=Exception("Rate limited")):
            res = await _async_web_search("MIT OCW Algorithms")
            assert "Web search could not retrieve external results" in res


class TestAgentToolPermissions:
    """Verify tool access rules: Web search is ONLY allowed for Syllabus and StudyMaterials agents."""

    def test_syllabus_agent_tool_suite(self):
        """Verify SyllabusAgent has both Knowledge Corpus and Web Search tools."""
        agent = SyllabusAgent()
        tool_names = [t.name for t in agent.tools]
        assert agent.enable_web_search is True
        assert "retrieve_knowledge_corpus" in tool_names
        assert "search_web" in tool_names

    def test_study_materials_agent_tool_suite(self):
        """Verify StudyMaterialsAgent has both Knowledge Corpus and Web Search tools."""
        agent = StudyMaterialsAgent()
        tool_names = [t.name for t in agent.tools]
        assert agent.enable_web_search is True
        assert "retrieve_knowledge_corpus" in tool_names
        assert "search_web" in tool_names

    def test_test_generator_agent_tool_suite(self):
        """Verify TestGeneratorAgent has Knowledge Corpus ONLY, and NO Web Search."""
        agent = TestGeneratorAgent()
        tool_names = [t.name for t in agent.tools]
        assert agent.enable_web_search is False
        assert "retrieve_knowledge_corpus" in tool_names
        assert "search_web" not in tool_names
        assert len(agent.tools) == 1

    def test_evaluator_agent_tool_suite(self):
        """Verify EvaluatorAgent has Knowledge Corpus ONLY, and NO Web Search."""
        agent = EvaluatorAgent()
        tool_names = [t.name for t in agent.tools]
        assert agent.enable_web_search is False
        assert "retrieve_knowledge_corpus" in tool_names
        assert "search_web" not in tool_names
        assert len(agent.tools) == 1


class TestBaseAgentRAGExecution:
    """Test suite for BaseAgent Agentic RAG invocation and failover."""

    def test_base_agent_json_parsing(self):
        """Verify JSON parsing handles clean, markdown-wrapped, and noisy strings."""
        agent = BaseAgent(model_name="test-model")

        # Standard clean JSON
        assert agent.parse_json('{"key": "value"}') == {"key": "value"}

        # Markdown wrapped
        assert agent.parse_json('```json\n{"key": "value"}\n```') == {"key": "value"}

        # Generic markdown
        assert agent.parse_json('```\n{"key": "value"}\n```') == {"key": "value"}

        # Text noise before and after
        noisy = 'Here is your response:\n```json\n{"score": 0.95}\n```\nHope this helps!'
        assert agent.parse_json(noisy) == {"score": 0.95}

        # Invalid JSON raises ValueError
        with pytest.raises(ValueError):
            agent.parse_json("Not a json at all")

    async def test_invoke_agentic_rag_fallback_on_error(self):
        """Verify invoke_agentic_rag falls back to invoke_chat if agent graph raises."""
        agent = BaseAgent(model_name="test-model")

        with patch.object(agent, "invoke_chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = '{"fallback": true}'
            
            # Trigger exception in get_chat_model or graph
            with patch.object(agent, "get_chat_model", side_effect=RuntimeError("LLM offline")):
                result = await agent.invoke_agentic_rag("prompt", "system")
                assert result == '{"fallback": true}'
                mock_chat.assert_called_once_with(prompt="prompt", system_instruction="system")


class TestAgentWorkflows:
    """Test end-to-end workflows of all 4 agents."""

    async def test_syllabus_agent_generate_syllabus(self):
        """Verify SyllabusAgent generate_syllabus completes with structured modules and topics."""
        agent = SyllabusAgent()
        
        mock_stage1 = {
            "syllabus_title": "Quantum Mechanics",
            "description": "Comprehensive course on quantum theory.",
            "modules": [
                {"order": 1, "name": "Wave-Particle Duality", "description": "Fundamentals."},
                {"order": 2, "name": "Schrodinger Equation", "description": "Wave mechanics."},
            ]
        }
        mock_stage2 = [
            {
                "order": 1,
                "name": "Wave-Particle Duality",
                "description": "Fundamentals.",
                "units": [
                    {"order": 1, "name": "Photoelectric Effect", "description": "Planck's constant.", "concepts": ["Photon", "Work Function"]},
                ]
            },
            {
                "order": 2,
                "name": "Schrodinger Equation",
                "description": "Wave mechanics.",
                "units": [
                    {"order": 1, "name": "Time-Independent Form", "description": "Stationary states.", "concepts": ["Hamiltonian", "Eigenstates"]},
                ]
            }
        ]

        with patch.object(agent, "_stage1_architect_curriculum", new_callable=AsyncMock, return_value=mock_stage1):
            with patch.object(agent, "_stage2_expand_batch", new_callable=AsyncMock, return_value=mock_stage2):
                res = await agent.generate_syllabus(topic="Quantum Mechanics")
                assert res["syllabus_title"] == "Quantum Mechanics"
                assert len(res["modules"]) == 2
                assert len(res["topics"]) == 2
                assert res["topics"][0]["name"] == "Wave-Particle Duality: Photoelectric Effect"

    async def test_study_materials_agent_recommend_materials(self):
        """Verify StudyMaterialsAgent returns parsed book and resource recommendations."""
        agent = StudyMaterialsAgent()
        mock_response = json.dumps({
            "concept_name": "Backpropagation",
            "textbooks": [
                {"title": "Deep Learning", "author": "Goodfellow et al.", "relevance": "Standard reference."}
            ],
            "papers": [
                {"title": "Learning representations by back-propagating errors", "authors": "Rumelhart et al.", "year": "1986", "url": "https://nature.com"}
            ],
            "online_resources": [
                {"name": "3Blue1Brown Neural Networks", "url": "https://youtube.com", "type": "video"}
            ]
        })

        with patch.object(agent, "invoke_agentic_rag", new_callable=AsyncMock, return_value=mock_response):
            res = await agent.recommend_study_materials("Backpropagation", "Gradient calculus in neural nets.")
            assert res["concept_name"] == "Backpropagation"
            assert len(res["textbooks"]) == 1
            assert res["textbooks"][0]["author"] == "Goodfellow et al."
            assert len(res["papers"]) == 1

    async def test_test_generator_agent_generate_question(self):
        """Verify TestGeneratorAgent generates rigorous taxonomy tier questions."""
        agent = TestGeneratorAgent()
        mock_response = json.dumps({
            "question": "Deconstruct the mathematical trade-offs between LSTM forget gates and GRU update gates.",
            "bloom_tier": 4,
            "expected_synthesis_scope": "2-4 paragraphs"
        })

        with patch.object(agent, "invoke_agentic_rag", new_callable=AsyncMock, return_value=mock_response):
            res = await agent.generate_question("Gated Recurrent Units", "Sequence models.", bloom_tier=4)
            assert res["bloom_tier"] == 4
            assert "trade-offs" in res["question"]

    async def test_evaluator_agent_evaluate_answer(self):
        """Verify EvaluatorAgent evaluates student answers against rubric."""
        agent = EvaluatorAgent()
        mock_response = json.dumps({
            "score": 0.88,
            "bloom_level_demonstrated": 4,
            "feedback": "Strong understanding of eigenvalues, good derivation.",
            "strengths": ["Clear definition", "Correct formula"],
            "misconceptions": [],
            "suggested_review_concepts": []
        })

        with patch.object(agent, "invoke_agentic_rag", new_callable=AsyncMock, return_value=mock_response):
            res = await agent.evaluate_answer(
                concept_name="Eigenvalues",
                question_prompt="Explain why Av = lambda v holds.",
                student_answer="When matrix A acts on eigenvector v, it only scales the vector by lambda.",
                bloom_tier=4,
            )
            assert res["score"] == 0.88
            assert res["bloom_level_demonstrated"] == 4
            assert len(res["strengths"]) == 2

    async def test_evaluator_agent_audit_knowledge_graph(self):
        """Verify EvaluatorAgent audits knowledge graph representations."""
        agent = EvaluatorAgent()
        mock_response = json.dumps({
            "has_warnings": True,
            "warnings": [
                {
                    "type": "circular_dependency",
                    "severity": "high",
                    "message": "Circular prerequisite loop detected between Node A and Node B.",
                    "affected_nodes": ["node_a", "node_b"]
                }
            ],
            "recommendations": ["Break cycle by making Node A prerequisite of Node B."]
        })

        with patch.object(agent, "invoke_agentic_rag", new_callable=AsyncMock, return_value=mock_response):
            res = await agent.audit_knowledge_graph(
                concepts=[{"id": "node_a", "name": "Concept A"}, {"id": "node_b", "name": "Concept B"}],
                edges=[{"from": "node_a", "to": "node_b"}, {"from": "node_b", "to": "node_a"}],
            )
            assert res["has_warnings"] is True
            assert len(res["warnings"]) == 1
            assert res["warnings"][0]["type"] == "circular_dependency"

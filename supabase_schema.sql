-- ===============================================================================
-- SUPERLEARN COGNITIVE ENGINE — SUPABASE RELATIONAL SCHEMA DDL
-- ===============================================================================
-- 
-- Run this script in the Supabase SQL Editor to initialize or verify all tables,
-- foreign keys, cascading constraints, and indexes.
-- 
-- Tables Created:
-- 1. classrooms
-- 2. cognitive_concepts
-- 3. cognitive_knowledge_edges
-- 4. classroom_syllabus_items
-- 5. cognitive_test_sessions
-- 6. classroom_mindmaps
-- 7. learner_cognitive_profiles
-- ===============================================================================

-- Enable UUID extension if not enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. CLASSROOMS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS classrooms (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    title VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    topic_query VARCHAR(255) NOT NULL,
    is_approved BOOLEAN NOT NULL DEFAULT FALSE,
    mindmap_image_url TEXT,
    mindmap_parsed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classrooms_title ON classrooms(title);
CREATE INDEX IF NOT EXISTS idx_classrooms_created_at ON classrooms(created_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. COGNITIVE CONCEPTS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cognitive_concepts (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    graph_partition VARCHAR(16) NOT NULL DEFAULT 'grand',
    position_x DOUBLE PRECISION,
    position_y DOUBLE PRECISION,
    bloom_level INTEGER NOT NULL DEFAULT 1,
    mastery_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    fsrs_stability DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    fsrs_difficulty DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    last_retrieval_timestamp TIMESTAMPTZ,
    total_review_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_concepts_classroom_id ON cognitive_concepts(classroom_id);
CREATE INDEX IF NOT EXISTS idx_concepts_name ON cognitive_concepts(name);
CREATE INDEX IF NOT EXISTS idx_concepts_partition ON cognitive_concepts(graph_partition);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. COGNITIVE KNOWLEDGE EDGES TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cognitive_knowledge_edges (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE,
    source_concept_id VARCHAR(36) NOT NULL REFERENCES cognitive_concepts(id) ON DELETE CASCADE,
    target_concept_id VARCHAR(36) NOT NULL REFERENCES cognitive_concepts(id) ON DELETE CASCADE,
    semantic_relation_label VARCHAR(255) NOT NULL DEFAULT 'prerequisite_for',
    graph_partition VARCHAR(16) NOT NULL DEFAULT 'grand',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edges_classroom_id ON cognitive_knowledge_edges(classroom_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON cognitive_knowledge_edges(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON cognitive_knowledge_edges(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_edges_partition ON cognitive_knowledge_edges(graph_partition);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. CLASSROOM SYLLABUS ITEMS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS classroom_syllabus_items (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    classroom_id VARCHAR(36) NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    order_index INTEGER NOT NULL DEFAULT 0,
    concept_id VARCHAR(36) NOT NULL REFERENCES cognitive_concepts(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_syllabus_classroom_id ON classroom_syllabus_items(classroom_id);
CREATE INDEX IF NOT EXISTS idx_syllabus_concept_id ON classroom_syllabus_items(concept_id);
CREATE INDEX IF NOT EXISTS idx_syllabus_order ON classroom_syllabus_items(classroom_id, order_index ASC);

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. COGNITIVE TEST SESSIONS TABLE (BLOOM ASSESSMENTS & SCORES)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cognitive_test_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE,
    concept_id VARCHAR(36) REFERENCES cognitive_concepts(id) ON DELETE SET NULL,
    bloom_tier INTEGER NOT NULL DEFAULT 4,
    generated_question_prompt TEXT NOT NULL,
    student_answer_text TEXT,
    evaluation_score DOUBLE PRECISION,
    qualitative_feedback TEXT,
    detected_misconceptions_json TEXT,
    strengths_json TEXT,
    suggested_review_json TEXT,
    effort_latency_seconds INTEGER,
    is_submitted BOOLEAN NOT NULL DEFAULT FALSE,
    session_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_submitted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_test_sessions_classroom_id ON cognitive_test_sessions(classroom_id);
CREATE INDEX IF NOT EXISTS idx_test_sessions_concept_id ON cognitive_test_sessions(concept_id);
CREATE INDEX IF NOT EXISTS idx_test_sessions_submitted ON cognitive_test_sessions(is_submitted, session_submitted_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. CLASSROOM MINDMAP UPLOADS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS classroom_mindmaps (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    classroom_id VARCHAR(36) NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    image_url TEXT,
    parsed_nodes_count INTEGER NOT NULL DEFAULT 0,
    parsed_edges_count INTEGER NOT NULL DEFAULT 0,
    extracted_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mindmaps_classroom_id ON classroom_mindmaps(classroom_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- 7. LEARNER COGNITIVE PROFILES TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS learner_cognitive_profiles (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total_test_sessions_completed INTEGER NOT NULL DEFAULT 0,
    exponential_moving_average_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    consecutive_streak_days INTEGER NOT NULL DEFAULT 0,
    last_active_timestamp TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Comments for Supabase Dashboard Schema UI
COMMENT ON TABLE classrooms IS 'Study spaces containing university curricula and persistent activity';
COMMENT ON TABLE cognitive_concepts IS 'Atomic concept nodes with FSRS memory stability and mastery metrics';
COMMENT ON TABLE cognitive_knowledge_edges IS 'Directed dependency and prerequisite edges in Grand and Mental Schema graphs';
COMMENT ON TABLE classroom_syllabus_items IS 'Sequential syllabus modules and lessons for each classroom';
COMMENT ON TABLE cognitive_test_sessions IS 'Anti-spoiler Bloom assessment attempts, forced-generation answers, and rubric evaluations';
COMMENT ON TABLE classroom_mindmaps IS 'Historical record of uploaded mind map images and extracted concept DAGs';

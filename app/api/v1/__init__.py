"""
===============================================================================
V1 API ROUTER AGGREGATION REGISTRY
===============================================================================
"""

from app.api.v1.dynamic_syllabus_router import router as dynamic_syllabus_router
from app.api.v1.multisource_contrast_router import router as multisource_contrast_router
from app.api.v1.mental_schema_router import router as mental_schema_router
from app.api.v1.bloom_assessment_router import router as bloom_assessment_router
from app.api.v1.syllabus_master_router import router as syllabus_master_router
from app.api.v1.study_materials_router import router as study_materials_router

__all__ = [
    "dynamic_syllabus_router",
    "multisource_contrast_router",
    "mental_schema_router",
    "bloom_assessment_router",
    "syllabus_master_router",
    "study_materials_router",
]


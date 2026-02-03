"""
Taxonomía y esquemas de datos para la clasificación de afiliaciones.

Garantiza que todos los backends (Gemini, Local) devuelvan exactamente
los mismos campos con tipos consistentes.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Literales para la taxonomía (valores permitidos)
OrgType = Literal[
    "supranational_organization",
    "government",
    "university",
    "research_institute",
    "company",
    "ngo",
    "hospital",
    "other",
    "unknown",
]

GovLevel = Literal[
    "federal",
    "state",
    "local",
    "unknown",
    "non_applicable",
]

GovLocalType = Literal[
    "city",
    "county",
    "other_local",
    "unknown",
    "non_applicable",
]

MissionResearchCategory = Literal[
    "NonResearch",
    "Enabler",
    "AppliedResearch",
    "AcademicResearch",
    "unknown",
]


class ClassificationResult(BaseModel):
    """
    Resultado de clasificación de una afiliación.

    Todos los backends (Gemini, Local) deben devolver exactamente estos campos.
    """

    org_type: OrgType = Field(..., description="Tipo de organización")
    gov_level: GovLevel = Field(..., description="Nivel de gobierno si aplica")
    gov_local_type: GovLocalType = Field(..., description="Tipo local si gov_level=local")
    mission_research_category: MissionResearchCategory = Field(
        ..., description="Categoría de misión investigadora"
    )
    mission_research: int = Field(..., ge=0, le=1, description="1 si tiene misión investigación, 0 si no")
    rationale: str = Field(default="", description="Justificación breve")

    # Campos opcionales para compatibilidad con reglas y CSV
    confidence_org_type: Optional[float] = None
    confidence_gov_level: Optional[float] = None
    confidence_mission_research: Optional[float] = None

    class Config:
        extra = "allow"  # Permite afid, ror_*, etc. al fusionar con otros datos

    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario para CSV y compatibilidad con código existente."""
        return self.model_dump()


def classification_result_from_dict(data: dict[str, Any]) -> ClassificationResult:
    """
    Construye un ClassificationResult desde un dict (p. ej. respuesta del LLM).
    Aplica valores por defecto para campos faltantes o inválidos.
    """
    valid_org_types: set[str] = {
        "supranational_organization", "government", "university", "research_institute",
        "company", "ngo", "hospital", "other", "unknown",
    }
    valid_gov_levels: set[str] = {"federal", "state", "local", "unknown", "non_applicable"}
    valid_gov_local: set[str] = {"city", "county", "other_local", "unknown", "non_applicable"}
    valid_mission: set[str] = {"NonResearch", "Enabler", "AppliedResearch", "AcademicResearch", "unknown"}

    org_type = data.get("org_type") or "unknown"
    if org_type not in valid_org_types:
        org_type = "unknown"

    gov_level = data.get("gov_level") or "non_applicable"
    if org_type != "government":
        gov_level = "non_applicable"
    elif gov_level not in valid_gov_levels:
        gov_level = "unknown"

    gov_local_type = data.get("gov_local_type") or "non_applicable"
    if org_type != "government" or gov_level != "local":
        gov_local_type = "non_applicable"
    elif gov_local_type not in valid_gov_local:
        gov_local_type = "unknown"

    mission_category = data.get("mission_research_category") or "unknown"
    if mission_category not in valid_mission:
        mission_category = "unknown"

    mission_research = data.get("mission_research")
    if mission_research not in (0, 1):
        mission_research = 1 if mission_category in ("AppliedResearch", "AcademicResearch") else 0

    return ClassificationResult(
        org_type=org_type,
        gov_level=gov_level,
        gov_local_type=gov_local_type,
        mission_research_category=mission_category,
        mission_research=int(mission_research),
        rationale=str(data.get("rationale", "") or ""),
        confidence_org_type=data.get("confidence_org_type"),
        confidence_gov_level=data.get("confidence_gov_level"),
        confidence_mission_research=data.get("confidence_mission_research"),
    )

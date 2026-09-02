from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_herg_prediction_website.py"
WEB_ROOT = Path(__file__).parents[1] / "web/herg"
LAUNCHER = Path(__file__).parents[1] / "scripts/launch_herg_prediction_website.sh"
LOCAL_LAUNCHER = Path(__file__).parents[1] / "scripts/launch_herg_prediction_website_local.sh"
STOPPER = Path(__file__).parents[1] / "scripts/stop_herg_prediction_website.sh"

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_herg_prediction_website", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_basic_authorization_is_valid() -> None:
    module = _module()
    value = module._basic_authorization("research", "secret")
    scheme, encoded = value.split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == "research:secret"


def test_probability_validation_rejects_nonfinite_and_out_of_range() -> None:
    module = _module()
    assert module._finite_probability(0.42) == 0.42
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(module.WebsiteError):
            module._finite_probability(value)


def test_prediction_mode_is_explicit_and_closed() -> None:
    module = _module()
    assert module._prediction_mode(None) == "ligand"
    assert module._prediction_mode("ligand") == "ligand"
    assert module._prediction_mode("ligand_receptor") == "ligand_receptor"
    with pytest.raises(module.WebsiteError):
        module._prediction_mode("receptor_only")

    assert module._feature_mode(None) == "atomwise"
    assert module._feature_mode("functional_group") == "functional_group"
    with pytest.raises(module.WebsiteError, match="Feature mode"):
        module._feature_mode("custom")

    assert module._boolean_flag(True, "Test flag") is True
    assert module._boolean_flag(False, "Test flag") is False
    with pytest.raises(module.WebsiteError, match="true or false"):
        module._boolean_flag("true", "Test flag")


def test_receptor_selection_is_ordered_bounded_and_closed() -> None:
    module = _module()
    assert module._receptor_selection(None) == ("8ZYO",)
    assert module._receptor_selection(["8zyn", "9CHQ"]) == ("8ZYN", "9CHQ")
    with pytest.raises(module.WebsiteError, match="JSON array"):
        module._receptor_selection("8ZYO")
    with pytest.raises(module.WebsiteError, match="between 1 and 6"):
        module._receptor_selection([])
    assert module._receptor_selection(list(module.RECEPTOR_STATES)) == tuple(module.RECEPTOR_STATES)
    with pytest.raises(module.WebsiteError, match="between 1 and 6"):
        module._receptor_selection([*module.RECEPTOR_STATES, "1ABC"])
    with pytest.raises(module.WebsiteError, match="only once"):
        module._receptor_selection(["8ZYO", "8zyo"])
    with pytest.raises(module.WebsiteError, match="Unsupported receptor"):
        module._receptor_selection(["1ABC"])


def test_static_assets_and_routes_are_closed() -> None:
    module = _module()
    module._validate_static_files(WEB_ROOT)
    assert set(module.STATIC_ROUTES) == {
        "/",
        "/index.html",
        "/assets/styles.css",
        "/assets/app.js",
    }


def test_site_copy_matches_promoted_policy() -> None:
    html = (WEB_ROOT / "index.html").read_text()
    normalized_html = " ".join(html.split())
    assert "hERG Prediction Platform" in html
    assert "IC10" in html and "IC30" in html and "IC50" in html
    assert "Ligand-only remains the default" in html
    assert "Ligand + receptor" in html
    assert "Compound Properties" in html
    assert "Property Explorer" in html
    assert "AutoDock Vina" in html
    assert "truly combines ligand and receptor features" in normalized_html
    assert "did not conclusively beat the strongest ligand-only comparator" in normalized_html
    assert "V14/V14.1 research preview" in html
    assert "Non-default · retrospective" in html
    assert "Global OOF-residual tier frequencies" in html
    assert "candidate − parent" in html
    assert "Generate candidates" in html
    assert "up to 100 bounded one-step edits" in html
    assert "Research use only" in html
    assert "Public comparator A · observed 43.353 µM" in html
    assert "Public comparator B · observed 11.493 µM" in html
    assert "Zenodo 8359714 dev corpus" in html
    assert "CID 16760153" in html and "CID 60196404" in html
    assert "chloride salt standardized to parent" in html
    assert "Internal validation" not in html
    assert "Internal potency" not in html
    assert "Private internal structures and measurements are" in normalized_html
    assert "Per-record analogue details withheld" in html
    assert "Nearest measured analogues" not in html
    assert "not an unbiased performance estimate" in normalized_html
    assert "disclosed hard failure" in normalized_html
    assert "representative audit runs" in normalized_html
    assert "Representative ligand + receptor IC50" in html
    assert "classification and IC50 regression are separate research models" in normalized_html
    assert "Select 1–6 Vina receptor states" in html
    assert "Functional-group augmented" in html
    assert "Expand heavy-molecule analysis" in html
    assert "Safe: IC50 &gt;30 µM; Moderate: 1 ≤ IC50 ≤ 30 µM; Potent: IC50 &lt;1 µM" in html
    assert "Fresh receptor runtime" in html
    assert "V10.3" not in html
    assert "Menin" not in html


def test_selected_evaluation_table_is_complete_and_transparent() -> None:
    html = (WEB_ROOT / "index.html").read_text()
    assert html.count('scope="row"') == 2
    assert "Exact training-overlap demonstration" in html
    assert "43.353" in html and "37.737" in html and "46.419" in html
    assert "11.493" in html and "12.343" in html and "11.547" in html


def test_static_site_contains_no_private_internal_validation_structures() -> None:
    html = (WEB_ROOT / "index.html").read_text()
    private_markers = (
        "COCC1=CC(N2CCC3(CC2)",
        "COC(=O)N[C@H]1CCC[C@@H]1[C@]",
        "Internal validation A",
        "Internal potency stress",
    )
    assert all(marker not in html for marker in private_markers)


def test_generation_mode_keeps_parent_enabled_and_hides_manual_candidate() -> None:
    html = (WEB_ROOT / "index.html").read_text()
    assert re.search(r'<div>\s*<label for="parent-smiles">', html)
    assert re.search(r'<div data-optimize-manual>\s*<label for="candidate-smiles">', html)


def test_prediction_includes_integrity_checked_nondefault_v14_preview() -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    prediction = predictor.predict("CCO")
    capability = predictor.info["capabilities"]["multi_receptor_diagnostics"]
    assert capability["request_field"] == "receptor_ids"
    assert capability["model_driving_receptors"] == ["8ZYO"]
    assert {row["id"] for row in capability["available_receptors"]} == {
        "8ZYN",
        "8ZYO",
        "8ZYP",
        "8ZYQ",
        "9CHP",
        "9CHQ",
    }
    assert all(row["prepared_assets_verified"] for row in capability["available_receptors"])
    mw_capability = predictor.info["capabilities"]["experimental_mw_sensitivity"]
    assert mw_capability["available"] is True
    assert mw_capability["default_enabled"] is False
    assert mw_capability["request_field"] == "mw_delta_stress_test"
    assert mw_capability["primary_prediction_unchanged"] is True
    assert set(predictor.receptor_assets) == {
        "8ZYN",
        "8ZYO",
        "8ZYP",
        "8ZYQ",
        "9CHP",
        "9CHQ",
    }
    preview = prediction["research_preview"]
    assert prediction["scientific_scope"]["ligand_only_default"] is True
    assert prediction["prediction_mode"] == "ligand"
    assert prediction["receptor_aware"]["status"] == "not_requested"
    assert prediction["scientific_scope"]["receptor_research_mode_available"] is True
    assert prediction["scientific_scope"]["receptor_features_used"] is False
    assert prediction["main_ic50"]["model_source"].endswith("RDKit2D+Morgan regression")
    assert prediction["structure_depiction"]["data_uri"].startswith("data:image/svg+xml;base64,")
    assert prediction["property_profile"]["descriptors"]["tpsa"]["method"] == ("RDKit Ertl-style TPSA")
    assert prediction["standardization"]["model_used_smiles"] == prediction["smiles"]
    assert "formal charges may be neutralized" in prediction["standardization"]["disclosure"]
    assert "heavy_molecule_diagnostic" in prediction
    consistency = prediction["classification_regression_consistency"]
    assert consistency["result_kind"] == "model_output_diagnostic"
    assert consistency["standalone_classifier"]["model"].endswith("RDKit2D+Morgan")
    assert "separately trained" in consistency["interpretation"]
    assert prediction["runtime"]["ligand_seconds"] >= 0
    assert prediction["scientific_scope"]["receptor_prediction_promoted"] is False
    assert prediction["applicability"]["nearest_analog_records_disclosed"] is False
    assert "nearest_analogs" not in prediction["applicability"]
    serialized_prediction = json.dumps(prediction)
    for private_key in ("structure_id", "observed_pic50", "observed_ic50_um"):
        assert private_key not in serialized_prediction
    assert preview["status"] == "research_preview_non_default"
    assert preview["receptor_model"]["promoted"] is False
    assert sum(preview["classification"]["probabilities"].values()) == pytest.approx(1.0)
    assert preview["validation"]["campaigns"] == 6
    assert preview["validation"]["structures"] == 1_224
    assert preview["validation"]["version"] == "V14 classification + V14.1 regression"
    assert preview["validation"]["regression"]["mae"] == pytest.approx(0.4759729911)
    assert (
        preview["validation"]["delta_vs_frozen"]["classification_balanced_accuracy"]["campaigns_better"] == 6
    )
    assert preview["validation"]["regression_delta_vs_v14_linear"]["ci95"][1] < 0

    functional = predictor.predict(
        "CCO",
        feature_mode="functional_group",
        heavy_molecule_analysis=True,
    )
    assert functional["feature_mode"]["selected"] == "functional_group"
    assert functional["feature_mode"]["validated_default_preserved"] is True
    assert functional["selected_quantitative_result"]["surface"] == "functional_group_research"
    assert functional["functional_group_research"]["status"] == "research_only_failed_promotion"
    assert functional["functional_group_research"]["validation"][
        "production_or_default_promotion_supported"
    ] is False
    assert functional["advanced_heavy_analysis"]["enabled"] is True
    assert functional["advanced_heavy_analysis"]["primary_prediction_unchanged"] is True


def test_private_internal_examples_are_exact_local_workbook_records() -> None:
    module = _module()
    result = module._load_internal_examples()
    assert result["result_kind"] == "local_private_measured_examples"
    assert len(result["examples"]) == 10
    assert [row["id"] for row in result["examples"]] == [
        f"MENIN-HERG-{index:02d}" for index in range(1, 11)
    ]
    assert {row["measurement"]["relation"] for row in result["examples"]} == {"=", "<", ">"}
    assert all(row["molecular_weight_da"] > 650 for row in result["examples"])
    assert "public tunnel" in result["privacy"]


def test_private_examples_endpoint_is_explicitly_local_launcher_only(tmp_path: Path) -> None:
    module = _module()

    class StubPredictor:
        info = {}

    private = module._load_internal_examples()
    enabled_handler = module._handler(StubPredictor(), tmp_path, "research", None, private)
    enabled = object.__new__(enabled_handler)
    enabled.path = "/api/internal-examples"
    enabled.headers = {}
    captured = {}
    enabled._json = lambda status, value, **_kwargs: captured.update(status=status, value=value)
    enabled._get(include_body=True)
    assert captured["status"] == module.HTTPStatus.OK
    assert len(captured["value"]["examples"]) == 10

    disabled_handler = module._handler(StubPredictor(), tmp_path, "research", None)
    disabled = object.__new__(disabled_handler)
    disabled.path = "/api/internal-examples"
    disabled.headers = {}
    captured = {}
    disabled._json = lambda status, value, **_kwargs: captured.update(status=status, value=value)
    disabled._get(include_body=True)
    assert captured["status"] == module.HTTPStatus.NOT_FOUND

    assert "--enable-internal-examples" in LOCAL_LAUNCHER.read_text()
    assert "--enable-internal-examples" not in LAUNCHER.read_text()


def test_properties_only_mode_is_method_labeled_and_model_free() -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    result = predictor.properties("CCO")
    assert result["analysis_mode"] == "properties_only"
    assert result["descriptors"]["clogp"]["method"] == "RDKit Wildman–Crippen MolLogP"
    assert result["descriptors"]["tpsa"]["method"] == "RDKit Ertl-style TPSA"
    assert len(result["descriptors"]) >= 15
    assert len(result["training_associations"]) == 5
    assert all(0.0 <= row["percentile"] <= 1.0 for row in result["training_associations"])
    assert result["structure_depiction"]["data_uri"].startswith("data:image/svg+xml;base64,")
    assert result["submitted_smiles"] == "CCO"
    assert result["standardization"]["model_used_smiles"] == result["smiles"]
    assert "standardized parent" in result["standardization"]["disclosure"]
    assert "main_ic50" not in result
    assert "not causal effects" in result["claim_boundary"]


def test_manual_comparison_is_ligand_only_separated_and_scientifically_labeled() -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    result = predictor.compare(
        "CN(C)CCc1ccccc1",
        "CC(=O)N(C)CCc1ccccc1",
        known_parent_ic50_um=8.0,
    )
    json.dumps(result, allow_nan=False)

    assert result["analysis_mode"] == "manual_parent_candidate_comparison"
    assert result["prediction_mode"] == "ligand"
    assert result["measured_values"]["parent_ic50_um"] == 8.0
    assert result["measured_values"]["parent_pic50"] == pytest.approx(5.096910013)
    assert result["model_outputs"]["result_kind"] == "model_prediction"
    assert result["calculated_properties"]["result_kind"] == "calculated_descriptors"
    assert result["heuristic_interpretation"]["result_kind"] == "heuristic_interpretation"
    assert result["model_outputs"]["parent"]["regression"]["predicted_ic50_um"] > 0
    assert set(result["model_outputs"]["parent"]["classifier"]["probabilities"]) == {
        "Safe",
        "Moderate",
        "Potent",
    }
    deltas = result["calculated_properties"]["deltas"]
    assert {
        "molecular_weight",
        "tpsa",
        "clogp",
        "hbd",
        "hba",
        "formal_charge",
        "rotatable_bonds",
        "basic_centers",
    } <= set(deltas)
    assert deltas["basic_centers"]["method"].endswith("not a pKa calculation")
    assert result["comparison"]["direction"] in {
        "improved",
        "worsened",
        "essentially_unchanged",
    }
    assert 0.0 <= result["comparison"]["similarity"]["morgan_tanimoto"] <= 1.0
    assert result["comparison"]["absolute_fold_change"] >= 1.0
    assert result["experimental_mw_sensitivity"] is None
    assert result["comparison"]["functional_group_changes"]["added"]
    assert "not SHAP attribution" in result["heuristic_interpretation"]["claim_boundary"]
    assert "No candidate was generated" in result["claim_boundary"]

    identical = predictor.compare("CCO", "CCO")
    assert identical["comparison"]["similarity"]["morgan_tanimoto"] == pytest.approx(1.0)
    assert identical["comparison"]["direction"] == "essentially_unchanged"
    assert identical["comparison"]["low_predicted_sensitivity"] is True
    assert any("High structural similarity" in warning for warning in identical["comparison"]["warnings"])

    stressed = predictor.compare(
        "CN(C)CCc1ccccc1",
        "CC(=O)N(C)CCc1ccccc1",
        mw_delta_stress_test=True,
    )
    diagnostic = stressed["experimental_mw_sensitivity"]
    assert diagnostic["enabled"] is True
    assert diagnostic["default_enabled"] is False
    assert diagnostic["primary_prediction_unchanged"]["candidate_pic50"] == pytest.approx(
        stressed["model_outputs"]["candidate"]["regression"]["predicted_pic50"]
    )
    assert diagnostic["stress_scenario"]["applied_multiplier"] >= 1.0
    assert diagnostic["stress_scenario"]["direction_preserved"] is True
    assert diagnostic["metrics"]["stress_candidate_ic50_um"] > 0
    assert "not a calibrated IC50 prediction" in diagnostic["claim_boundary"]


def test_manual_comparison_rejects_empty_invalid_and_bad_measured_values() -> None:
    module = _module()
    predictor = object.__new__(module.WebsitePredictor)

    with pytest.raises(module.WebsiteError, match="parent SMILES"):
        predictor.compare("", "CCO")
    with pytest.raises(module.WebsiteError, match="candidate SMILES"):
        predictor.compare("CCO", "  ")
    with pytest.raises(module.WebsiteError, match="parent SMILES string could not be parsed"):
        predictor.compare("not-a-smiles", "CCO")
    with pytest.raises(module.WebsiteError, match="candidate SMILES string could not be parsed"):
        predictor.compare("CCO", "C1(")
    for bad_value in (0, -1, float("nan"), True, "unknown"):
        with pytest.raises(module.WebsiteError, match="positive number in µM"):
            predictor.compare("CCO", "CCN", bad_value)


def test_manual_comparison_small_change_and_heavy_diagnostics_are_explicit() -> None:
    module = _module()
    assert module.WebsitePredictor._morgan_tanimoto("CCO", "CCO") == pytest.approx(1.0)
    assert 0.0 <= module.WebsitePredictor._morgan_tanimoto("CCO", "CCN") <= 1.0

    predictor = object.__new__(module.WebsitePredictor)
    predictor.property_reference = [None] * 18_801
    properties = {
        "descriptors": {"molecular_weight": {"value": 800.0}},
        "training_associations": [{"key": "molecular_weight", "percentile": 0.99}],
        "docking_eligibility": {
            "eligible_under_current_receptor_model": False,
            "reason": "molecular_weight_gt_750",
        },
    }
    prediction = {
        "applicability": {
            "domain_label": "Extrapolative chemistry",
            "maximum_train_tanimoto": 0.42,
            "exact_training_overlap": False,
            "interpretation": "No close training analogue exists.",
        }
    }
    diagnostic = predictor._heavy_molecule_diagnostic(properties, prediction)
    assert diagnostic["above_current_receptor_750_da_limit"] is True
    assert diagnostic["at_or_above_empirical_95th_percentile"] is True
    assert diagnostic["training_reference_structures"] == 18_801
    assert any("ligand-only prediction completed" in row for row in diagnostic["warnings"])


def test_docking_execution_is_separate_from_receptor_model_applicability() -> None:
    module = _module()
    registry = pd.DataFrame(
        [
            {
                "docking_eligible": False,
                "docking_exclusion_reason": (
                    "molecular_weight_gt_750;heavy_atoms_gt_55;rotatable_bonds_gt_15"
                ),
                "docking_parent_smiles": "CCCC",
            }
        ]
    )
    contract = module._docking_execution_contract(registry)
    assert contract["docking_execution_eligible"] is True
    assert contract["receptor_feature_computation_eligible"] is True
    assert contract["receptor_model_applicable"] is False
    assert contract["final_receptor_prediction_eligible"] is False
    assert "molecular_weight_gt_750" in contract["advisory_model_domain_reasons"]
    execution = module._docking_execution_registry(registry)
    assert bool(execution.iloc[0].docking_eligible)

    unsupported = registry.copy()
    unsupported.loc[0, "docking_exclusion_reason"] = "unsupported_elements_U"
    blocked = module._docking_execution_contract(unsupported)
    assert blocked["docking_execution_eligible"] is False
    assert module._docking_execution_registry(unsupported).empty


def test_out_of_domain_8zyo_can_remain_a_docking_only_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    predictor = object.__new__(module.WebsitePredictor)
    predictor.receptor_bundle = {"vina_protocol": {"exhaustiveness": 8, "modes": 9}}
    registry = pd.DataFrame(
        [
            {
                "docking_eligible": False,
                "docking_exclusion_reason": "molecular_weight_gt_750",
                "docking_parent_smiles": "CCCC",
            }
        ]
    )

    def should_not_run_model(*_args):
        raise AssertionError("out-of-domain receptor model must not run")

    monkeypatch.setattr(predictor, "_receptor_prediction", should_not_run_model)
    monkeypatch.setattr(
        predictor,
        "_dock_diagnostic_receptors",
        lambda _registry, receptor_ids: {
            "status": "completed",
            "per_receptor": [
                {
                    "receptor_id": receptor_id,
                    "status": "completed",
                    "model_role": "model_driving_and_docking_diagnostic",
                    "model_prediction_generated": False,
                    "best_vina_affinity_kcal_mol": -6.5,
                }
                for receptor_id in receptor_ids
            ],
            "runtime_seconds": 1.0,
        },
    )
    panel = predictor._receptor_panel_prediction({}, registry, pd.DataFrame(), ("8ZYO",))
    assert panel["receptor_aware"]["status"] == "unavailable_out_of_model_domain"
    assert panel["multi_receptor_diagnostics"]["status"] == "completed"
    assert panel["multi_receptor_diagnostics"]["docking_only_receptors"] == ["8ZYO"]
    assert panel["multi_receptor_diagnostics"]["model_driving_receptors"] == []
    assert panel["multi_receptor_diagnostics"]["per_receptor"][0][
        "model_prediction_generated"
    ] is False


def test_compare_endpoint_requires_authentication_and_forwards_clean_payload(tmp_path: Path) -> None:
    module = _module()

    class StubPredictor:
        info = {}

        def compare(
            self,
            parent_smiles,
            candidate_smiles,
            known_parent_ic50_um=None,
            mw_delta_stress_test=False,
        ):
            return {
                "parent_smiles": parent_smiles,
                "candidate_smiles": candidate_smiles,
                "known_parent_ic50_um": known_parent_ic50_um,
                "mw_delta_stress_test": mw_delta_stress_test,
            }

    handler = module._handler(StubPredictor(), tmp_path, "research", "secret")
    body = json.dumps(
        {
            "parent_smiles": "CCO",
            "candidate_smiles": "CCN",
            "known_parent_ic50_um": 8.0,
            "mw_delta_stress_test": True,
        }
    ).encode()
    request = object.__new__(handler)
    request.path = "/api/compare"
    request.headers = {}
    assert request._authorized(request.path) is False

    request.headers = {
        "Authorization": module._basic_authorization("research", "secret"),
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    request.rfile = io.BytesIO(body)
    captured = {}
    request._json = lambda status, value: captured.update(status=status, value=value)
    assert request._authorized(request.path) is True
    request.do_POST()
    assert captured["status"] == module.HTTPStatus.OK
    assert captured["value"] == {
        "parent_smiles": "CCO",
        "candidate_smiles": "CCN",
        "known_parent_ic50_um": 8.0,
        "mw_delta_stress_test": True,
    }


def test_generate_endpoint_forwards_bounded_request(tmp_path: Path) -> None:
    module = _module()

    class StubPredictor:
        info = {}

        def generate_candidates(self, parent_smiles, max_candidates=100):
            return {"parent_smiles": parent_smiles, "max_candidates": max_candidates}

    handler = module._handler(StubPredictor(), tmp_path, "research", None)
    body = json.dumps({"parent_smiles": "CCO", "max_candidates": 7}).encode()
    request = object.__new__(handler)
    request.path = "/api/generate"
    request.headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    request.rfile = io.BytesIO(body)
    captured = {}
    request._json = lambda status, value: captured.update(status=status, value=value)
    request.do_POST()
    assert captured["status"] == module.HTTPStatus.OK
    assert captured["value"] == {"parent_smiles": "CCO", "max_candidates": 7}


def test_post_rejection_drains_small_body_before_reusing_connection(tmp_path: Path) -> None:
    module = _module()

    class StubPredictor:
        info = {}

    handler = module._handler(StubPredictor(), tmp_path, "research", None)
    body = b"{}"
    request = object.__new__(handler)
    request.path = "/api/predict"
    request.headers = {
        "Content-Type": "text/plain",
        "Content-Length": str(len(body)),
    }
    request.rfile = io.BytesIO(body)
    captured = {}
    request._json = lambda status, value: captured.update(status=status, value=value)
    request.do_POST()

    assert captured == {
        "status": module.HTTPStatus.BAD_REQUEST,
        "value": {"error": "Send the prediction request as JSON"},
    }
    assert request.rfile.read() == b""


def test_experimental_generation_uses_frozen_predictions_and_evidence_ranking() -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    result = predictor.generate_candidates(
        "CCN(CC)CCOc1ccc(C(=O)NCCc2ccc(F)cc2)cc1",
        5,
    )
    assert result["analysis_mode"] == "experimental_bounded_candidate_generation"
    assert result["generation"]["requested_candidate_count"] == 5
    assert result["generation"]["successfully_evaluated_count"] == 5
    assert [row["ranking"]["rank"] for row in result["ranked_candidates"]] == [1, 2, 3, 4, 5]
    for row in result["ranked_candidates"]:
        assert row["generation"]["lineages"]
        assert row["model_output"]["regression"]["interval90_um"]["lower"] > 0
        assert row["model_output"]["applicability"]["domain_label"]
        assert "not a probability" in row["ranking"]["score_boundary"]
    assert "never added to training data" in result["claim_boundary"]


def test_predict_endpoint_forwards_canonical_receptor_ids(tmp_path: Path) -> None:
    module = _module()

    class StubPredictor:
        info = {}

        def predict(
            self,
            smiles,
            mode="ligand",
            receptor_ids=None,
            feature_mode="atomwise",
            heavy_molecule_analysis=False,
        ):
            return {
                "smiles": smiles,
                "mode": mode,
                "receptor_ids": receptor_ids,
                "feature_mode": feature_mode,
                "heavy_molecule_analysis": heavy_molecule_analysis,
            }

    handler = module._handler(StubPredictor(), tmp_path, "research", None)
    body = json.dumps(
        {
            "smiles": "CCO",
            "mode": "ligand_receptor",
            "receptor_ids": ["8ZYN", "8ZYO"],
        }
    ).encode()
    request = object.__new__(handler)
    request.path = "/api/predict"
    request.headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    request.rfile = io.BytesIO(body)
    captured = {}
    request._json = lambda status, value: captured.update(status=status, value=value)
    request.do_POST()

    assert captured["status"] == module.HTTPStatus.OK
    assert captured["value"]["receptor_ids"] == ["8ZYN", "8ZYO"]
    assert captured["value"]["feature_mode"] == "atomwise"
    assert captured["value"]["heavy_molecule_analysis"] is False


def test_receptor_mode_uses_receptor_result_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    calls = 0

    def receptor_stub(result, registry, ligand):
        nonlocal calls
        calls += 1
        assert result["classification"]["model"].endswith("RDKit2D+Morgan")
        assert bool(registry.iloc[0].docking_eligible)
        assert len(ligand) == 1
        return {
            "status": "research_only_non_default",
            "definition": "ligand features plus receptor features",
            "runtime": {"total_seconds": 1.0, "cache_hit": False},
        }

    monkeypatch.setattr(predictor, "_receptor_prediction", receptor_stub)
    fresh = predictor.predict("CCO", "ligand_receptor")
    cached = predictor.predict("CCO", "ligand_receptor")

    assert calls == 1
    assert fresh["prediction_mode"] == "ligand_receptor"
    assert fresh["scientific_scope"]["receptor_features_used"] is True
    assert fresh["runtime"]["receptor_cache_hit"] is False
    assert cached["runtime"]["receptor_cache_hit"] is True
    assert cached["receptor_aware"]["runtime"]["cache_hit"] is True


def test_requested_but_unavailable_receptor_mode_does_not_claim_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )

    monkeypatch.setattr(
        predictor,
        "_receptor_prediction",
        lambda result, registry, ligand: {
            "status": "unavailable",
            "reason": "molecular_weight_gt_750",
        },
    )
    prediction = predictor.predict("CCO", "ligand_receptor")

    assert prediction["prediction_mode"] == "ligand_receptor"
    assert prediction["scientific_scope"]["receptor_mode_requested"] is True
    assert prediction["scientific_scope"]["receptor_features_used"] is False
    assert prediction["receptor_aware"]["status"] == "unavailable"


def test_selected_receptor_panel_keeps_only_8zyo_model_driving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    model_calls = 0
    diagnostic_calls = []

    def receptor_stub(result, registry, ligand):
        nonlocal model_calls
        model_calls += 1
        return {
            "status": "research_only_non_default",
            "docking": {
                "best_affinity_kcal_mol": -8.1,
                "median_affinity_kcal_mol": -7.4,
                "returned_pose_count": 9,
                "ligand_efficiency_kcal_mol_per_heavy_atom": -0.31,
                "best_pose_contacts": {"T623": 1, "S624": 2, "S649": 0, "Y652": 3, "F656": 2},
            },
            "runtime": {"total_seconds": 1.0, "cache_hit": False},
        }

    def diagnostics_stub(registry, receptor_ids):
        diagnostic_calls.append(receptor_ids)
        return {
            "status": "completed",
            "per_receptor": [
                {
                    "receptor_id": receptor_id,
                    "status": "completed",
                    "model_role": "docking_only_diagnostic",
                    "best_vina_affinity_kcal_mol": -7.0,
                }
                for receptor_id in receptor_ids
            ],
            "runtime_seconds": 2.0,
        }

    monkeypatch.setattr(predictor, "_receptor_prediction", receptor_stub)
    monkeypatch.setattr(predictor, "_dock_diagnostic_receptors", diagnostics_stub)
    selected = ["8ZYN", "8ZYO", "8ZYP", "8ZYQ", "9CHP", "9CHQ"]
    result = predictor.predict("CCO", "ligand_receptor", selected)

    assert model_calls == 1
    assert diagnostic_calls == [("8ZYN", "8ZYP", "8ZYQ", "9CHP", "9CHQ")]
    assert result["selected_receptor_ids"] == selected
    diagnostics = result["multi_receptor_diagnostics"]
    assert diagnostics["model_driving_receptors"] == ["8ZYO"]
    assert diagnostics["docking_only_receptors"] == ["8ZYN", "8ZYP", "8ZYQ", "9CHP", "9CHQ"]
    assert [row["receptor_id"] for row in diagnostics["per_receptor"]] == selected
    assert diagnostics["per_receptor"][1]["model_prediction_generated"] is True
    assert result["scientific_scope"]["receptor_features_used"] is True
    assert result["scientific_scope"]["receptor_docking_diagnostics_used"] is True


def test_non_8zyo_panel_is_docking_only_and_cache_depends_on_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    predictor = module.WebsitePredictor(
        module.DEFAULT_V101,
        module.DEFAULT_V102,
        module.DEFAULT_V103,
        module.DEFAULT_V14,
        module.DEFAULT_V141,
    )
    panel_calls = []

    def panel_stub(result, registry, ligand, receptor_ids):
        panel_calls.append(receptor_ids)
        return {
            "receptor_aware": {
                "status": "docking_only_no_model",
                "reason": "8ZYO was not selected",
            },
            "multi_receptor_diagnostics": {
                "status": "completed",
                "requested_receptors": list(receptor_ids),
                "per_receptor": [
                    {"receptor_id": receptor_id, "status": "completed"} for receptor_id in receptor_ids
                ],
                "cache_hit": False,
            },
        }

    monkeypatch.setattr(predictor, "_receptor_panel_prediction", panel_stub)
    first = predictor.predict("CCN", "ligand_receptor", ["8ZYN", "9CHP"])
    cached = predictor.predict("CCN", "ligand_receptor", ["8ZYN", "9CHP"])
    reordered = predictor.predict("CCN", "ligand_receptor", ["9CHP", "8ZYN"])

    assert panel_calls == [("8ZYN", "9CHP"), ("9CHP", "8ZYN")]
    assert first["scientific_scope"]["receptor_features_used"] is False
    assert first["receptor_aware"]["status"] == "docking_only_no_model"
    assert first["scientific_scope"]["receptor_docking_diagnostics_used"] is True
    assert cached["runtime"]["receptor_cache_hit"] is True
    assert reordered["runtime"]["receptor_cache_hit"] is False


def test_client_avoids_dynamic_html_injection() -> None:
    javascript = (WEB_ROOT / "app.js").read_text()
    assert ".innerHTML" not in javascript
    assert ".style." not in javascript
    assert "textContent" in javascript
    assert 'document.createElement("progress")' in javascript
    assert "AbortController" in javascript
    assert "Download CSV" in (WEB_ROOT / "index.html").read_text()


def test_client_element_references_exist_and_html_ids_are_unique() -> None:
    html = (WEB_ROOT / "index.html").read_text()
    javascript = (WEB_ROOT / "app.js").read_text()
    html_ids = re.findall(r'\bid="([^"]+)"', html)
    client_ids = set(re.findall(r'element\("([^"]+)"\)', javascript))
    assert len(html_ids) == len(set(html_ids))
    assert client_ids <= set(html_ids)


def test_private_launcher_requires_auth_and_verifies_the_tunnel() -> None:
    launcher = LAUNCHER.read_text()
    stopper = STOPPER.read_text()
    assert "HERG_WEBSITE_PASSWORD" in launcher
    assert "openssl rand" in launcher
    assert '"$SERVER_SCRIPT" validate' in launcher
    assert 'HTTP_STATUS" != "401"' in launcher
    assert 'LOCAL_HTTP_STATUS" != "401"' in launcher
    assert "curl --fail --silent --connect-timeout 3 --max-time 5" in launcher
    assert '--user "$AUTH_USERNAME:$AUTH_PASSWORD" "$PUBLIC_URL/"' in launcher
    assert "PUBLIC_AUTH_VERIFIED" in launcher
    assert "SSH_BIN" in launcher
    assert "-p 443" in launcher
    assert '-R "0:127.0.0.1:$PORT"' in launcher
    assert "free.pinggy.io" in launcher
    assert "free\\.pinggy\\.net" in launcher
    assert "run\\.pinggy-free\\.link" in launcher
    assert "ALTERNATE_URL" in launcher
    assert "StrictHostKeyChecking=accept-new" in launcher
    assert 'UserKnownHostsFile="$KNOWN_HOSTS"' in launcher
    assert 'while kill -0 "$SERVER_PID"' in launcher
    assert "expected_marker" in stopper
    assert '"free.pinggy.io"' in stopper
    assert "launch_info.txt" in stopper
    assert "HERG_ENABLE_PUBLIC_TUNNEL" in launcher
    assert "disabled by default" in launcher


def test_local_launcher_never_creates_a_tunnel() -> None:
    launcher = LOCAL_LAUNCHER.read_text()
    assert 'serve --host 127.0.0.1 --port "$PORT"' in launcher
    assert "localhost only; no tunnel was created" in launcher
    assert "free.pinggy.io" not in launcher
    assert "HERG_ENABLE_PUBLIC_TUNNEL" not in launcher

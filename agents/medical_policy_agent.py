import json
import os
from typing import Dict, Any
import requests
from datetime import datetime

class MedicalPolicyAgent:
    def __init__(self, policy_source: str = "local_policy.json"):
        """Initialize with a source for Bajaj policy rules (local file or URL)."""
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        self.policy_source = policy_source
        self.policy_rules = self._load_policy_rules()
        self.name = "MedicalPolicyAgent"

    def _load_policy_rules(self) -> Dict[str, Any]:
        """Load policy rules from a file, URL, or fallback to defaults."""
        default_rules = {
            "coverage_limits": {
                "hospitalization": 500000,
                "pre_existing": 100000,
                "outpatient": 20000,
                "maternity": 30000,
                "sum_insured_max": 5000000
            },
            "pre_post_hospitalization": {
                "pre_days": 60,
                "post_days": 90
            },
            "exclusions": [
                "Cosmetic surgery",
                "Experimental treatments",
                "Self-inflicted injuries",
                "HIV/AIDS",
                "Non-medical expenses"
            ],
            "claim_process": {
                "submission_deadline": 30,
                "cashless_approval_time": 60,
                "pre_authorization": True,
                "free_look_period": 30
            },
            "network_hospitals": True,
            "claim_settlement_ratio": 0.9064
        }

        valid_urls = [
            "https://www.policybazaar.com/insurance-companies/bajaj-allianz-health-insurance/",
            "https://www.bajajallianz.com/health-insurance-plans/private-health-insurance.html"
        ]
        if self.policy_source in valid_urls:
            try:
                print(f"Attempting to fetch policy data from {self.policy_source} at {self.last_updated}")
                response = requests.get(self.policy_source, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                print("Warning: Direct JSON fetch from URL not supported. Using default rules.")
                return default_rules
            except requests.RequestException as e:
                print(f"Failed to fetch policy: {e}. Using default rules.")
                return default_rules
        else:
            try:
                if os.path.exists(self.policy_source):
                    with open(self.policy_source, 'r') as f:
                        return json.load(f)
                print(f"Policy file {self.policy_source} not found. Using default rules.")
                return default_rules
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {self.policy_source}: {e}. Using default rules.")
                return default_rules
            except Exception as e:
                print(f"Error loading policy file: {e}. Using default rules.")
                return default_rules

    def evaluate_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a medical claim against Bajaj policy rules with dynamic amount validation."""
        response = {"decision": "pending", "reason": [], "details": {}}

        try:
            claim_amount = claim_data.get("amount", 0)
            claim_type = claim_data.get("type", "").lower()
            condition = claim_data.get("condition", "").lower()
            is_pre_existing = claim_data.get("pre_existing", False)
            is_planned = claim_data.get("planned_treatment", False)
            submitted_days = claim_data.get("submitted_days", 0)
            pre_hosp_days = claim_data.get("pre_hosp_days", 0)
            post_hosp_days = claim_data.get("post_hosp_days", 0)
            web_info = claim_data.get("web_info", [])

            # Attempt to determine max_limit from web_info or policy_source
            max_limit = self.policy_rules["coverage_limits"].get(claim_type, self.policy_rules["coverage_limits"]["sum_insured_max"])
            approved_amount = claim_amount

            # Check web_info for dynamic limit (simplified simulation)
            if web_info:
                for info in web_info:
                    if isinstance(info, dict) and "coverage_limit" in info:
                        dynamic_limit = info.get("coverage_limit", 0)
                        if dynamic_limit > 0:
                            max_limit = dynamic_limit
                            break

            if is_pre_existing:
                limit = self.policy_rules["coverage_limits"].get("pre_existing", 0)
                if claim_amount > limit:
                    approved_amount = limit
                    response["reason"].append(f"Claim is for pre-existing condition. Limit is ₹{limit}.")
                    response["decision"] = "partially approved"

            if "outpatient" in claim_type or "opd" in condition:
                limit = self.policy_rules["coverage_limits"].get("outpatient", 0)
                if claim_amount > limit:
                    approved_amount = limit
                    response["reason"].append(f"Outpatient claim capped at ₹{limit}.")
                    response["decision"] = "partially approved"

            if "maternity" in condition:
                limit = self.policy_rules["coverage_limits"].get("maternity", 0)
                if claim_amount > limit:
                    approved_amount = limit
                    response["reason"].append(f"Maternity coverage limited to ₹{limit}.")
                    response["decision"] = "partially approved"

            if any(ex.lower() in condition for ex in self.policy_rules["exclusions"]):
                response["decision"] = "denied"
                response["reason"].append(f"Claim involves excluded condition: {condition}.")

            if claim_type == "hospitalization":
                if pre_hosp_days > self.policy_rules["pre_post_hospitalization"]["pre_days"]:
                    response["decision"] = "denied"
                    response["reason"].append(f"Exceeds {self.policy_rules['pre_post_hospitalization']['pre_days']} days pre-hospitalization coverage.")
                if post_hosp_days > self.policy_rules["pre_post_hospitalization"]["post_days"]:
                    response["decision"] = "denied"
                    response["reason"].append(f"Exceeds {self.policy_rules['pre_post_hospitalization']['post_days']} days post-hospitalization coverage.")

            if submitted_days > self.policy_rules["claim_process"]["submission_deadline"]:
                response["decision"] = "denied"
                response["reason"].append(f"Claim submitted after {self.policy_rules['claim_process']['submission_deadline']} days.")
            if is_planned and not claim_data.get("pre_authorized", False):
                response["decision"] = "denied"
                response["reason"].append("Pre-authorization required for planned treatment.")

            if claim_amount > max_limit:
                approved_amount = max_limit
                response["reason"].append(f"Claim amount ₹{claim_amount} exceeds policy limit of ₹{max_limit}. Approving ₹{max_limit}.")
                response["decision"] = "partially approved"
            elif max_limit <= 0:
                response["reason"].append("Unable to determine policy limit from available data. Please contact policy provider for clarification.")
                response["decision"] = "pending"

            if response["decision"] == "pending":
                response["decision"] = "approved"

            response["details"] = {
                "claim_amount": claim_amount,
                "approved_amount": approved_amount,
                "claim_type": claim_type,
                "condition": condition,
                "is_pre_existing": is_pre_existing,
                "policy_limits": self.policy_rules["coverage_limits"],
                "dynamic_limit": max_limit,
                "exclusions_applied": any(ex.lower() in condition for ex in self.policy_rules["exclusions"]),
                "submission_days": submitted_days,
                "pre_hosp_days": pre_hosp_days,
                "post_hosp_days": post_hosp_days,
                "cashless_eligible": self.policy_rules["network_hospitals"],
                "claim_settlement_ratio": self.policy_rules["claim_settlement_ratio"]
            }

        except KeyError as e:
            response["decision"] = "error"
            response["reason"].append(f"Missing claim data: {str(e)}")
        except Exception as e:
            response["decision"] = "error"
            response["reason"].append(f"Unexpected error: {str(e)}")

        return response

    def explain_decision(self, evaluation: Dict[str, Any]) -> str:
        """Generate a detailed explanation of the claim decision."""
        if evaluation["decision"] == "approved":
            return f"🎉 Claim APPROVED! Amount: ₹{evaluation['details']['approved_amount']} approved for {evaluation['details']['claim_type']}. " \
                   f"Meets policy limit. Pre/Post days within range. (Settlement ratio: {evaluation['details']['claim_settlement_ratio']*100:.2f}%)"
        elif evaluation["decision"] == "partially approved":
            return f"🟡 Claim PARTIALLY APPROVED. ₹{evaluation['details']['approved_amount']} approved out of ₹{evaluation['details']['claim_amount']}. " \
                   f"Reason(s): {', '.join(evaluation.get('reason', []))}"
        elif evaluation["decision"] == "denied":
            return f"❌ Claim DENIED. Reason(s): {', '.join(evaluation.get('reason', []))}"
        else:
            return f"🤔 Error evaluating claim. Reason: {', '.join(evaluation.get('reason', []))}"

    def process_claim(self, claim_data: str) -> Dict[str, Any]:
        """Process a JSON string claim and return evaluation with explanation."""
        try:
            claim = json.loads(claim_data)
            evaluation = self.evaluate_claim(claim)
            evaluation["explanation"] = self.explain_decision(evaluation)
            return evaluation
        except json.JSONDecodeError:
            return {"decision": "error", "reason": ["Invalid JSON input"], "explanation": "Please provide valid claim data!"}


# Test the agent
if __name__ == "__main__":
    agent = MedicalPolicyAgent("https://www.policybazaar.com/insurance-companies/bajaj-allianz-health-insurance/")
    sample_claim = {
        "amount": 600000,
        "type": "hospitalization",
        "condition": "appendicitis",
        "pre_existing": False,
        "planned_treatment": False,
        "submitted_days": 15,
        "pre_hosp_days": 30,
        "post_hosp_days": 45,
        "pre_authorized": True,
        "web_info": [{"coverage_limit": 700000}]  # Simulated retrieved data
    }
    result = agent.process_claim(json.dumps(sample_claim))
    print(f"Agent: {agent.name}")
    print(f"Last Updated: {agent.last_updated}")
    print(json.dumps(result, indent=2))
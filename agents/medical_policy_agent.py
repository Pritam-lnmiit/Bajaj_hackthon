"""Medical Policy Agent: The Guru of Bajaj Medical Insurance Claims!"""

import json
import os
from typing import Dict, Any, Optional
import requests
from datetime import datetime

class MedicalPolicyAgent:
    def __init__(self, policy_source: str = "local_policy.json"):
        """Initialize with a source for Bajaj policy rules (local file or URL).
        
        Args:
            policy_source: Path to local file (default: local_policy.json) or URL
                          (e.g., https://www.policybazaar.com/insurance-companies/bajaj-allianz-health-insurance/
                          or https://www.bajajallianz.com/health-insurance-plans/private-health-insurance.html).
        """
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")  # Initialize timestamp
        self.policy_source = policy_source
        self.policy_rules = self._load_policy_rules()
        self.name = "MedicalPolicyAgent"

    def _load_policy_rules(self) -> Dict[str, Any]:
        """Load policy rules from a file, URL, or fallback to defaults."""
        # Default rules based on Bajaj Allianz and Policybazaar data
        default_rules = {
            "coverage_limits": {
                "hospitalization": 500000,  # ₹5 lakh base (up to ₹50 lakh per Bajaj site)
                "pre_existing": 100000,     # ₹1 lakh (waiting period applies)
                "outpatient": 20000,        # ₹20k for OPD (estimated)
                "maternity": 30000,         # ₹30k (estimated, plan-specific)
                "sum_insured_max": 5000000  # ₹50 lakh max (Bajaj private plans)
            },
            "pre_post_hospitalization": {
                "pre_days": 60,             # 60 days pre-hospitalization (Bajaj)
                "post_days": 90             # 90 days post-hospitalization (Bajaj)
            },
            "exclusions": [
                "Cosmetic surgery",
                "Experimental treatments",
                "Self-inflicted injuries",
                "HIV/AIDS",
                "Non-medical expenses"      # Bajaj-specific exclusion
            ],
            "claim_process": {
                "submission_deadline": 30,  # 30 days (Bajaj free-look period hint)
                "cashless_approval_time": 60,  # 60 minutes (Policybazaar)
                "pre_authorization": True,  # Required for planned (Bajaj)
                "free_look_period": 30      # 30-day free-look (Bajaj)
            },
            "network_hospitals": True,       # Cashless at 6,500+ network hospitals (Bajaj)
            "claim_settlement_ratio": 0.9064  # 90.64% (Policybazaar FY 2021-22)
        }

        # Check if the source is one of the provided links
        valid_urls = [
            "https://www.policybazaar.com/insurance-companies/bajaj-allianz-health-insurance/",
            "https://www.bajajallianz.com/health-insurance-plans/private-health-insurance.html"
        ]
        if self.policy_source in valid_urls:
            try:
                print(f"Attempting to fetch policy data from {self.policy_source} at {self.last_updated}")
                response = requests.get(self.policy_source, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                # Note: Real scraping requires parsing HTML; this assumes a JSON API (not available here)
                print(f"Warning: Direct JSON fetch from {self.policy_source} not supported. Please provide a JSON file or API.")
                return default_rules
            except requests.RequestException as e:
                print(f"Failed to fetch policy from {self.policy_source}: {e}. Falling back to defaults.")
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
                print(f"Error loading policy file {self.policy_source}: {e}. Using default rules.")
                return default_rules

    def evaluate_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a medical claim against Bajaj policy rules."""
        response = {"decision": "pending", "reason": [], "details": {}}

        try:
            # Extract claim details
            claim_amount = claim_data.get("amount", 0)
            claim_type = claim_data.get("type", "").lower()
            condition = claim_data.get("condition", "").lower()
            is_pre_existing = claim_data.get("pre_existing", False)
            is_planned = claim_data.get("planned_treatment", False)
            submitted_days = claim_data.get("submitted_days", 0)
            pre_hosp_days = claim_data.get("pre_hosp_days", 0)
            post_hosp_days = claim_data.get("post_hosp_days", 0)

            # Check coverage limits
            max_limit = self.policy_rules["coverage_limits"].get(claim_type, self.policy_rules["coverage_limits"]["sum_insured_max"])
            if claim_amount > max_limit:
                response["decision"] = "denied"
                response["reason"].append(f"Exceeds {claim_type} limit of ₹{max_limit}.")
            else:
                response["decision"] = "approved"

            # Check pre/post-hospitalization
            if claim_type == "hospitalization":
                if pre_hosp_days > self.policy_rules["pre_post_hospitalization"]["pre_days"]:
                    response["decision"] = "denied"
                    response["reason"].append(f"Exceeds {self.policy_rules['pre_post_hospitalization']['pre_days']} days pre-hospitalization coverage.")
                if post_hosp_days > self.policy_rules["pre_post_hospitalization"]["post_days"]:
                    response["decision"] = "denied"
                    response["reason"].append(f"Exceeds {self.policy_rules['pre_post_hospitalization']['post_days']} days post-hospitalization coverage.")

            # Check exclusions
            if any(exclusion.lower() in condition for exclusion in self.policy_rules["exclusions"]):
                response["decision"] = "denied"
                response["reason"].append(f"Claim involves excluded condition: {condition}.")

            # Check process compliance
            if submitted_days > self.policy_rules["claim_process"]["submission_deadline"]:
                response["decision"] = "denied"
                response["reason"].append(f"Claim submitted after {self.policy_rules['claim_process']['submission_deadline']} days.")
            if is_planned and not claim_data.get("pre_authorized", False):
                response["decision"] = "denied"
                response["reason"].append("Pre-authorization required for planned treatment.")

            # Add details for explanation
            response["details"] = {
                "claim_amount": claim_amount,
                "claim_type": claim_type,
                "condition": condition,
                "policy_limits": self.policy_rules["coverage_limits"],
                "exclusions_applied": any(exclusion.lower() in condition for exclusion in self.policy_rules["exclusions"]),
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
            return f"🎉 Claim APPROVED! Amount: ₹{evaluation['details']['claim_amount']} for {evaluation['details']['claim_type']} " \
                   f"meets Bajaj policy limits (up to ₹{evaluation['details']['policy_limits'].get(evaluation['details']['claim_type'], evaluation['details']['policy_limits']['sum_insured_max'])}). " \
                   f"Pre/Post: {evaluation['details']['pre_hosp_days']}/{evaluation['details']['post_hosp_days']} days ok. " \
                   f"Cashless available at 6,500+ network hospitals. Process complied! (Settlement ratio: {evaluation['details']['claim_settlement_ratio']*100}%)"
        elif evaluation["decision"] == "denied":
            return f"😱 Claim DENIED! Reasons: {', '.join(evaluation['reason'])}. " \
                   f"Details: Amount ₹{evaluation['details']['claim_amount']} for {evaluation['details']['claim_type']} " \
                   f"vs limit ₹{evaluation['details']['policy_limits'].get(evaluation['details']['claim_type'], evaluation['details']['policy_limits']['sum_insured_max'])}. " \
                   f"Contact Bajaj at 1800-209-5858!"
        else:
            return f"🤔 Error in evaluation: {', '.join(evaluation['reason'])}. Contact Bajaj support at 1800-209-5858!"

    def process_claim(self, claim_data: str) -> Dict[str, Any]:
        """Process a JSON string claim and return evaluation with explanation."""
        try:
            claim = json.loads(claim_data)
            evaluation = self.evaluate_claim(claim)
            evaluation["explanation"] = self.explain_decision(evaluation)
            return evaluation
        except json.JSONDecodeError:
            return {"decision": "error", "reason": ["Invalid JSON input"], "explanation": "Please provide valid claim data!"}

# Example usage (for testing)
if __name__ == "__main__":
    # Test with one of the links or local file
    agent = MedicalPolicyAgent("https://www.policybazaar.com/insurance-companies/bajaj-allianz-health-insurance/")
    # Alternatively, use the other link: MedicalPolicyAgent("https://www.bajajallianz.com/health-insurance-plans/private-health-insurance.html")
    # Or stick with local: MedicalPolicyAgent("local_policy.json")
    sample_claim = {
        "amount": 400000,
        "type": "hospitalization",
        "condition": "appendicitis",
        "pre_existing": False,
        "planned_treatment": False,
        "submitted_days": 15,
        "pre_hosp_days": 30,
        "post_hosp_days": 45,
        "pre_authorized": True
    }
    result = agent.process_claim(json.dumps(sample_claim))
    print(f"Agent: {agent.name}")
    print(f"Last Updated: {agent.last_updated}")
    print(json.dumps(result, indent=2))
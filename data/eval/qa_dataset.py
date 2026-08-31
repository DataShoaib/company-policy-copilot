"""53 curated HR policy Q&A pairs, checked against data/policies/*.md directly.
Each item also carries category/question_type/difficulty so retrieval and
generation quality can be scored per-slice, not just as one blended average."""

from dataclasses import dataclass, field, asdict


@dataclass
class QAItem:
    id: str
    question: str
    ground_truth: str
    source_doc: str  # filename in data/policies/, "multiple" for multi-hop, "none" for unanswerable
    category: str  # leave | compensation | conduct | performance | recruitment | finance | it | cross-policy
    question_type: str  # factual | numeric | multi_hop | paraphrase | exact_keyword | unanswerable
    difficulty: str  # easy | medium | hard
    expected_keywords: list = field(default_factory=list)  # substrings a correctly-retrieved chunk must contain


QA_ITEMS = [
    # ---------------------------------------------------------------- LEAVE
    QAItem(
        id="leave-01",
        question="How many days of Casual Leave am I entitled to per year?",
        ground_truth="Employees are entitled to 12 days of Casual Leave (CL) per calendar year, accrued at 1 day per month and credited on the 1st of each month.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="numeric",
        difficulty="easy",
        expected_keywords=["Casual Leave", "12 days"],
    ),
    QAItem(
        id="leave-02",
        question="Can I carry forward my unused Casual Leave to next year?",
        ground_truth="No. Carry forward is not allowed for Casual Leave; any unused CL lapses on December 31st.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="factual",
        difficulty="easy",
        expected_keywords=["Carry forward", "NOT allowed", "lapses"],
    ),
    QAItem(
        id="leave-03",
        question="How many weeks of maternity leave do I get for my first child?",
        ground_truth="26 weeks (182 days) of paid maternity leave for the first two children, with up to 8 weeks allowed before the expected delivery date. Pay is 100% of last drawn salary.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="numeric",
        difficulty="medium",
        expected_keywords=["Maternity Leave", "26 weeks", "182 days"],
    ),
    QAItem(
        id="leave-04",
        question="mera baby hone wala hai, third child, kitni maternity leave milegi?",
        ground_truth="For the third child onwards, maternity leave entitlement is 12 weeks (reduced from the 26 weeks given for the first two children).",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="paraphrase",
        difficulty="hard",
        expected_keywords=["third child", "12 weeks"],
    ),
    QAItem(
        id="leave-05",
        question="If I'm still on probation, how much sick leave and casual leave can I take?",
        ground_truth="During the 6-month probation period, employees are eligible only for pro-rated Casual Leave (6 days) and pro-rated Sick Leave (5 days). Earned Leave accrual only begins after probation is completed.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="multi_hop",
        difficulty="hard",
        expected_keywords=["Probation Period", "6 days", "5 days", "pro-rated"],
    ),
    QAItem(
        id="leave-06",
        question="What percentage of my Sick Leave balance can I encash when I resign?",
        ground_truth="50% of the accumulated Sick Leave balance can be encashed at retirement or resignation.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="numeric",
        difficulty="medium",
        expected_keywords=["Sick Leave", "50%", "Encashment"],
    ),
    QAItem(
        id="leave-07",
        question="How many days of paternity leave are male employees eligible for, and within what timeframe must it be taken?",
        ground_truth="15 days of paid paternity leave, to be taken within 30 days of the childbirth or adoption.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="factual",
        difficulty="easy",
        expected_keywords=["Paternity Leave", "15 days", "30 days"],
    ),
    QAItem(
        id="leave-08",
        question="What happens if I take more than 10 days of Leave Without Pay in a year?",
        ground_truth="More than 10 LWP days in a year triggers a performance review, and LWP also affects the annual increment calculation on a pro-rated basis.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="factual",
        difficulty="medium",
        expected_keywords=["Leave Without Pay", "10 LWP days", "performance review"],
    ),

    # --------------------------------------------------------- COMPENSATION
    QAItem(
        id="comp-01",
        question="What is the employer's Provident Fund contribution percentage?",
        ground_truth="The employer contributes 12% of Basic Salary to PF, split as 8.33% to EPS and 3.67% to EPF, and this employer contribution is tax-free up to ₹7,500 per month.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="numeric",
        difficulty="medium",
        expected_keywords=["Employer Contribution", "12% of Basic", "8.33%", "3.67%"],
    ),
    QAItem(
        id="comp-02",
        question="How is gratuity calculated and what is the maximum tax-free amount?",
        ground_truth="Gratuity is calculated as (Last drawn Basic Salary x 15 x Years of Service) / 26. It requires a minimum of 5 years of continuous service and is tax-free up to ₹20 lakhs.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="numeric",
        difficulty="hard",
        expected_keywords=["Gratuity", "15 x Years of Service", "26", "20 lakhs"],
    ),
    QAItem(
        id="comp-03",
        question="Am I eligible for my annual performance bonus if I'm serving my notice period?",
        ground_truth="No. Employees serving a notice period at the time of bonus payment are NOT eligible for the performance bonus, and employees on a PIP are also not eligible.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="factual",
        difficulty="medium",
        expected_keywords=["notice period", "NOT eligible", "PIP"],
    ),
    QAItem(
        id="comp-04",
        question="What is the health insurance sum insured and who is covered under it?",
        ground_truth="The base health insurance policy covers the employee, spouse, and up to 2 children (up to age 25), with a sum insured of ₹5 lakhs per family per year. A top-up of up to ₹20 lakhs is available with shared premium.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="numeric",
        difficulty="medium",
        expected_keywords=["Health Insurance", "₹5 lakhs", "Spouse", "2 children"],
    ),
    QAItem(
        id="comp-05",
        question="I'm getting promoted from L2 to L3 next quarter — what salary increase should I expect, combining promotion and performance rules?",
        ground_truth="A cross-band promotion like L2 to L3 carries a minimum 20% increase per the salary revision policy, and separately the general promotion increment policy specifies 15-20% for standard promotions — so as a cross-band move, expect at least a 20% increase.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="multi_hop",
        difficulty="hard",
        expected_keywords=["Cross-band promotion", "20%", "Promotion Increment"],
    ),
    QAItem(
        id="comp-06",
        question="mujhe apni salary slip 15 tareek ko join karne pe kitni milegi is month?",
        ground_truth="Salary on joining is paid pro-rata using the formula (Monthly CTC / Number of days in month) x Days worked. For example, joining on the 15th of March would pay for 17 days (15th to 31st).",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="paraphrase",
        difficulty="hard",
        expected_keywords=["pro-rata", "Monthly CTC", "Days worked"],
    ),

    # --------------------------------------------------------------- CONDUCT
    QAItem(
        id="conduct-01",
        question="Who do I contact if I want to file a POSH / sexual harassment complaint?",
        ground_truth="Complaints should be submitted in writing to the Internal Complaints Committee (ICC) within 90 days of the incident. The ICC Chairperson is Priya Mehta (priya.mehta.icc@techcorp.in).",
        source_doc="03_code_of_conduct_policy.md",
        category="conduct",
        question_type="factual",
        difficulty="easy",
        expected_keywords=["ICC", "90 days", "Priya Mehta"],
    ),
    QAItem(
        id="conduct-02",
        question="Is sharing confidential company data with a competitor grounds for immediate termination?",
        ground_truth="Yes. Sharing confidential data with competitors is classified as Gross Misconduct, which is eligible for immediate termination without notice.",
        source_doc="03_code_of_conduct_policy.md",
        category="conduct",
        question_type="factual",
        difficulty="easy",
        expected_keywords=["Gross Misconduct", "Immediate Termination", "confidential data"],
    ),
    QAItem(
        id="conduct-03",
        question="What are the 5 steps of the progressive disciplinary process, in order?",
        ground_truth="1) Verbal Warning, 2) Written Warning (Show Cause Notice), 3) Performance Improvement Plan (PIP), 4) Suspension (with or without pay), 5) Termination.",
        source_doc="03_code_of_conduct_policy.md",
        category="conduct",
        question_type="factual",
        difficulty="medium",
        expected_keywords=["Verbal Warning", "Written Warning", "PIP", "Suspension", "Termination"],
    ),
    QAItem(
        id="conduct-04",
        question="If I have a grievance my manager can't resolve, what's the full escalation path?",
        ground_truth="Level 1: immediate manager (3 working days). Level 2: HR Business Partner (7 working days). Level 3: HR Director (14 working days). Level 4: Ombudsman / External Mediator for unresolved cases.",
        source_doc="03_code_of_conduct_policy.md",
        category="conduct",
        question_type="multi_hop",
        difficulty="medium",
        expected_keywords=["Level 1", "Level 2", "Level 3", "Level 4", "Ombudsman"],
    ),
    QAItem(
        id="conduct-05",
        question="How long is the non-compete clause after I leave, and does it apply to everyone?",
        ground_truth="The non-compete clause prevents joining direct competitors for 6 months post-exit, but it only applies to employees at L4 and above.",
        source_doc="03_code_of_conduct_policy.md",
        category="conduct",
        question_type="numeric",
        difficulty="medium",
        expected_keywords=["non-compete", "6 months", "L4 and above"],
    ),

    # ---------------------------------------------------------- PERFORMANCE
    QAItem(
        id="perf-01",
        question="What performance rating triggers a mandatory PIP?",
        ground_truth="A rating of 1 (Does Not Meet Expectations) triggers a PIP, and so do two consecutive ratings of 2 (Partially Meets Expectations).",
        source_doc="04_performance_management_policy.md",
        category="performance",
        question_type="factual",
        difficulty="easy",
        expected_keywords=["PIP Triggers", "Rating of 1", "two consecutive ratings of 2"],
    ),
    QAItem(
        id="perf-02",
        question="How long can a Performance Improvement Plan last?",
        ground_truth="A PIP can last from 30 days (critical cases) to 90 days (standard cases).",
        source_doc="04_performance_management_policy.md",
        category="performance",
        question_type="numeric",
        difficulty="easy",
        expected_keywords=["PIP", "30 days", "90 days"],
    ),
    QAItem(
        id="perf-03",
        question="What are the minimum eligibility requirements to be considered for a promotion?",
        ground_truth="Minimum 18 months in the current grade, at least 2 consecutive years of rating 3 or above (or 1 year with rating 4/5), and no active PIP or disciplinary action in the last 24 months.",
        source_doc="04_performance_management_policy.md",
        category="performance",
        question_type="multi_hop",
        difficulty="hard",
        expected_keywords=["18 months", "2 consecutive years", "rating 3", "24 months"],
    ),
    QAItem(
        id="perf-04",
        question="I got a rating of 4 this year — what performance bonus percentage of target should I expect?",
        ground_truth="A rating of 4 (Exceeds Expectations) corresponds to 120% of the target bonus.",
        source_doc="04_performance_management_policy.md",
        category="performance",
        question_type="numeric",
        difficulty="medium",
        expected_keywords=["Exceeds Expectations", "120%"],
    ),
    QAItem(
        id="perf-05",
        question="Can a manager rate more than a quarter of their team at the top two rating levels?",
        ground_truth="No. Managers cannot rate more than 25% of their team at ratings 4 or 5 combined, without explicit HR justification, due to the bell-curve calibration policy.",
        source_doc="04_performance_management_policy.md",
        category="performance",
        question_type="factual",
        difficulty="medium",
        expected_keywords=["25% of team", "4 or 5", "HR justification"],
    ),

    # --------------------------------------------------------- RECRUITMENT
    QAItem(
        id="recruit-01",
        question="How many interview rounds does an L3-level candidate go through?",
        ground_truth="L3 candidates go through 4 rounds: Screen + 2 Technical rounds + HR + Manager round.",
        source_doc="05_recruitment_onboarding_policy.md",
        category="recruitment",
        question_type="numeric",
        difficulty="easy",
        expected_keywords=["L3", "4 rounds", "Screen"],
    ),
    QAItem(
        id="recruit-02",
        question="What is the referral bonus for successfully referring someone hired at L4, and when is it paid?",
        ground_truth="₹50,000, paid after 6 months of the referral joining.",
        source_doc="05_recruitment_onboarding_policy.md",
        category="recruitment",
        question_type="numeric",
        difficulty="easy",
        expected_keywords=["L4", "₹50,000", "6 months"],
    ),
    QAItem(
        id="recruit-03",
        question="Can I refer my spouse for an open position and claim the referral bonus?",
        ground_truth="No. Employees cannot refer immediate family members, including spouse, parents, or siblings, under the Employee Referral Program.",
        source_doc="05_recruitment_onboarding_policy.md",
        category="recruitment",
        question_type="factual",
        difficulty="easy",
        expected_keywords=["Cannot refer immediate family", "spouse"],
    ),
    QAItem(
        id="recruit-04",
        question="How long does Background Verification take, and what happens if something adverse is found after I've already joined?",
        ground_truth="BGV takes 15-20 business days. If adverse findings emerge after the candidate has already joined, the employment is terminated (or the offer is revoked if pre-joining).",
        source_doc="05_recruitment_onboarding_policy.md",
        category="recruitment",
        question_type="multi_hop",
        difficulty="medium",
        expected_keywords=["15-20 business days", "adverse BGV", "terminated"],
    ),
    QAItem(
        id="recruit-05",
        question="What mandatory trainings must I complete in my first 30 days as a new joiner?",
        ground_truth="Information Security Awareness (2 hours), POSH Awareness (1 hour), Code of Conduct Certification (1 hour), and Data Privacy & GDPR Basics (1 hour) — all on the LMS.",
        source_doc="05_recruitment_onboarding_policy.md",
        category="recruitment",
        question_type="factual",
        difficulty="medium",
        expected_keywords=["Information Security Awareness", "POSH Awareness", "Code of Conduct Certification", "GDPR"],
    ),

    # -------------------------------------------------------------- FINANCE
    QAItem("finance-01", "What is the domestic daily meal allowance?", "The domestic meal allowance is INR 1,200 per day.", "06_finance_expense_policy.md", "finance", "numeric", "easy", ["INR 1,200"]),
    QAItem("finance-02", "When must I submit a business travel expense claim?", "Expense claims must be submitted within 30 calendar days after returning.", "06_finance_expense_policy.md", "finance", "numeric", "easy", ["30 calendar days"]),
    QAItem("finance-03", "Who approves an expense above INR 25,000?", "Expenses above INR 25,000 require manager and Finance approval before purchase.", "06_finance_expense_policy.md", "finance", "factual", "medium", ["INR 25,000", "Finance approval"]),

    # ------------------------------------------------------------------- IT
    QAItem("it-01", "How quickly must I report a lost company laptop?", "A lost device must be reported to IT Security within 24 hours.", "07_it_security_policy.md", "it", "numeric", "easy", ["within 24 hours"]),
    QAItem("it-02", "Is VPN required when accessing internal systems remotely?", "Yes. Remote access requires the company VPN and multi-factor authentication.", "07_it_security_policy.md", "it", "factual", "easy", ["company VPN", "multi-factor authentication"]),

    # ---------------------------------------------------------------- LEGAL
    QAItem("legal-01", "Who must review a vendor contract before it is signed?", "Vendor agreements must be sent to Legal before signing, and only authorized signatories may execute them.", "08_legal_compliance_policy.md", "legal", "factual", "easy", ["Legal", "authorized signatories"]),
    QAItem("legal-02", "How long must business records be retained?", "Business records must be retained for 7 years unless Legal gives a different instruction.", "08_legal_compliance_policy.md", "legal", "numeric", "easy", ["7 years"]),

    # ------------------------------------------------------------ OPERATIONS
    QAItem("operations-01", "How early should I submit a normal supply request?", "Operational supply requests must be submitted at least 5 business days before delivery is required.", "09_operations_workplace_policy.md", "operations", "numeric", "easy", ["5 business days"]),
    QAItem("operations-02", "What should employees do when there is a critical operational incident?", "Employees must follow the incident commander instructions and use the approved status channel.", "09_operations_workplace_policy.md", "operations", "factual", "medium", ["incident commander", "approved status channel"]),

    # ----------------------------------------------------------- CROSS-POLICY
    QAItem(
        id="cross-01",
        question="If I'm on a PIP right now, does that affect my eligibility for a promotion, an internal job transfer, or my performance bonus?",
        ground_truth="Yes to all three. An active PIP disqualifies you from promotion eligibility (Performance Policy), you cannot apply to Internal Job Postings without being off an active PIP (Recruitment Policy), and PIP employees are explicitly not eligible for the performance bonus (Compensation Policy).",
        source_doc="multiple",
        category="cross-policy",
        question_type="multi_hop",
        difficulty="hard",
        expected_keywords=["PIP", "promotion", "Internal Job Postings", "not eligible", "bonus"],
    ),
    QAItem(
        id="cross-02",
        question="A candidate is hired at L4 through a referral and the offer requires background verification — walk me through the referral bonus timing and the offer approval authority.",
        ground_truth="For an L4 hire: offer approval authority is HR Director + Department Head (Recruitment Policy), and the referral bonus of ₹50,000 is paid after 6 months of the referral's joining, contingent on BGV having cleared (15-20 business days).",
        source_doc="multiple",
        category="cross-policy",
        question_type="multi_hop",
        difficulty="hard",
        expected_keywords=["L4", "HR Director", "₹50,000", "6 months", "BGV"],
    ),

    # ------------------------------------------------- EXACT-TERM / CODE LOOKUP
    # These probe BM25's edge: rare lexical tokens (codes, IDs, form numbers)
    # that dense embeddings cannot reliably disambiguate from generic prose.
    # item 10 (HRP-001) uses a code that ALREADY existed in the corpus --
    # the argument for BM25/hybrid predates any data augmentation.
    QAItem(
        id="exact-leave-01",
        question="Which HRMS request type do I use to claim a compensatory off?",
        ground_truth="Comp-Off must be claimed in the ESS portal using HRMS request type CO-WKD-1; the legacy paper form FRM-LV-CO has been deprecated.",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="exact_keyword",
        difficulty="easy",
        expected_keywords=["CO-WKD-1", "FRM-LV-CO"],
    ),
    QAItem(
        id="exact-comp-01",
        question="What is the FBP plan code I need to select during flexible benefit plan enrollment?",
        ground_truth="FBP-FLEX-2026 must be selected on the HRMS Benefits tab when enrolling in cafeteria components under the Flexible Benefit Plan.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="exact_keyword",
        difficulty="easy",
        expected_keywords=["FBP-FLEX-2026"],
    ),
    QAItem(
        id="exact-comp-02",
        question="What is the group mediclaim master policy number referenced on insurance claims?",
        ground_truth="The Group Mediclaim master policy number is GRPMED-IN-4412, referenced on all insurance claims.",
        source_doc="02_compensation_payroll_policy.md",
        category="compensation",
        question_type="exact_keyword",
        difficulty="medium",
        expected_keywords=["GRPMED-IN-4412"],
    ),
    QAItem(
        id="exact-fin-01",
        question="Which expense category code applies to a client dinner reimbursement?",
        ground_truth="Client entertainment (business meals with external clients) must be tagged with expense category code EXP-ENT-05.",
        source_doc="06_finance_expense_policy.md",
        category="finance",
        question_type="exact_keyword",
        difficulty="medium",
        expected_keywords=["EXP-ENT-05"],
    ),
    QAItem(
        id="exact-fin-02",
        question="I need my monthly broadband bill reimbursed — which expense category code do I tag it with?",
        ground_truth="Telecom and internet bills are tagged with expense category code EXP-TEL-03.",
        source_doc="06_finance_expense_policy.md",
        category="finance",
        question_type="exact_keyword",
        difficulty="medium",
        expected_keywords=["EXP-TEL-03"],
    ),
    QAItem(
        id="exact-it-01",
        question="What service request code do I use on the IT self-service portal to get approval for installing new software?",
        ground_truth="New software requests are raised through the IT self-service portal using service request code IT-SW-117, reviewed by the IT Security team within 5 business days.",
        source_doc="07_it_security_policy.md",
        category="it",
        question_type="exact_keyword",
        difficulty="medium",
        expected_keywords=["IT-SW-117"],
    ),
    QAItem(
        id="exact-it-02",
        question="Which VPN access group do contractors connect through?",
        ground_truth="Contractors connect through the restricted VPN access group VPN-CNT-1, which does not grant access to source-code repositories.",
        source_doc="07_it_security_policy.md",
        category="it",
        question_type="exact_keyword",
        difficulty="hard",
        expected_keywords=["VPN-CNT-1"],
    ),
    QAItem(
        id="exact-rec-01",
        question="What format does a manpower requisition number follow?",
        ground_truth="Requisition numbers follow the format REQ-<LEVEL>-<DEPT>-YYYY (e.g., REQ-L4-ENG-2026) and must be quoted in all follow-up correspondence.",
        source_doc="05_recruitment_onboarding_policy.md",
        category="recruitment",
        question_type="exact_keyword",
        difficulty="medium",
        expected_keywords=["REQ-", "-ENG-2026"],
    ),
    QAItem(
        id="exact-docid-01",
        question="Under which document ID is the Leave Management Policy filed?",
        ground_truth="The Leave Management Policy carries Document ID HRP-001 (Version 2.1, effective January 1, 2024).",
        source_doc="01_leave_policy.md",
        category="leave",
        question_type="exact_keyword",
        difficulty="easy",
        expected_keywords=["HRP-001"],
    ),

    # ----------------------------------------------------------- UNANSWERABLE
    QAItem(
        id="unans-01",
        question="What is the company's remote work stipend for employees working from Dubai?",
        ground_truth="This is not answerable from the available policy documents. There is no international remote-work or Dubai-specific stipend defined in the Leave, Compensation, Conduct, Performance, or Recruitment policies provided.",
        source_doc="none",
        category="cross-policy",
        question_type="unanswerable",
        difficulty="hard",
        expected_keywords=[],
    ),
    QAItem(
        id="unans-02",
        question="What is the CEO's current base salary?",
        ground_truth="This is not answerable from the available policy documents. Individual executive compensation is not disclosed in these policy documents.",
        source_doc="none",
        category="cross-policy",
        question_type="unanswerable",
        difficulty="medium",
        expected_keywords=[],
    ),
    QAItem(
        id="unans-03",
        question="Does the company provide a car lease scheme for L2 employees?",
        ground_truth="This is not answerable from the available policy documents. The Compensation Policy only defines a fuel reimbursement benefit for L4 and above; no car lease scheme is described anywhere in the provided policies.",
        source_doc="none",
        category="cross-policy",
        question_type="unanswerable",
        difficulty="medium",
        expected_keywords=[],
    ),
]


# flat {question: [...], answer: [...]} shape for RAGAS Dataset.from_dict
dataset = {
    "question": [item.question for item in QA_ITEMS],
    "answer": [item.ground_truth for item in QA_ITEMS],
}

qa_records = [asdict(item) for item in QA_ITEMS]


def get_subset(question_type: str = None, category: str = None, difficulty: str = None):
    items = QA_ITEMS
    if question_type:
        items = [i for i in items if i.question_type == question_type]
    if category:
        items = [i for i in items if i.category == category]
    if difficulty:
        items = [i for i in items if i.difficulty == difficulty]
    return items


if __name__ == "__main__":
    print(f"Total QA items: {len(QA_ITEMS)}")
    from collections import Counter
    print("By type:", Counter(i.question_type for i in QA_ITEMS))
    print("By category:", Counter(i.category for i in QA_ITEMS))
    print("By difficulty:", Counter(i.difficulty for i in QA_ITEMS))

(define (domain south-asia-clean-cooking)
  (:requirements :strips :typing)
  (:types
    unit period coverage-tier budget-level
  )

  (:predicates
    ;; Infrastructure readiness (initial vs built in period ?p)
    (lpg-init-ready ?u - unit)
    (grid-init-ready ?u - unit)
    (lpg-infra-built-at ?u - unit ?p - period)
    (grid-infra-built-at ?u - unit ?p - period)
    (lpg-infra-built ?u - unit)
    (grid-infra-built ?u - unit)

    ;; Coverage state and linear tier progression
    (cov-tier ?u - unit ?c - coverage-tier)
    (next-cov ?c1 ?c2 - coverage-tier)
    (clean-fuel-deployed ?u - unit)

    ;; Sequential period ordering & strictly-earlier prerequisite relation
    (active-period ?p - period)
    (next-period ?p1 ?p2 - period)
    (strictly-before ?p1 ?p2 - period)

    ;; Per-unit-per-period rollout guard (respects per-period adoption cap)
    (slot-open ?u - unit ?p - period)

    ;; Ordered discrete budget levels (avoids unordered-token symmetry blow-up)
    (budget-level ?b - budget-level)
    (next-budget ?b1 ?b2 - budget-level)
  )

  ;; Build LPG infrastructure in active period ?p (unlocks strictly later periods)
  (:action build-lpg-infra
    :parameters (?u - unit ?p - period ?b1 ?b2 - budget-level)
    :precondition (and
      (active-period ?p)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (lpg-infra-built-at ?u ?p)
      (lpg-infra-built ?u)
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Build electric cooking grid infrastructure in active period ?p
  (:action build-grid-infra
    :parameters (?u - unit ?p - period ?b1 ?b2 - budget-level)
    :precondition (and
      (active-period ?p)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (grid-infra-built-at ?u ?p)
      (grid-infra-built ?u)
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Rollout LPG subsidy in initially ready unit (urban)
  (:action rollout-lpg-init-ready
    :parameters (?u - unit ?p - period ?c1 ?c2 - coverage-tier ?b1 ?b2 - budget-level)
    :precondition (and
      (lpg-init-ready ?u)
      (active-period ?p)
      (slot-open ?u ?p)
      (cov-tier ?u ?c1)
      (next-cov ?c1 ?c2)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (not (cov-tier ?u ?c1))
      (cov-tier ?u ?c2)
      (not (slot-open ?u ?p))
      (clean-fuel-deployed ?u)
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Rollout LPG subsidy in unready unit after infra built in a STRICTLY earlier period
  (:action rollout-lpg-after-infra
    :parameters (?u - unit ?p_infra ?p - period ?c1 ?c2 - coverage-tier ?b1 ?b2 - budget-level)
    :precondition (and
      (lpg-infra-built-at ?u ?p_infra)
      (strictly-before ?p_infra ?p)
      (active-period ?p)
      (slot-open ?u ?p)
      (cov-tier ?u ?c1)
      (next-cov ?c1 ?c2)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (not (cov-tier ?u ?c1))
      (cov-tier ?u ?c2)
      (not (slot-open ?u ?p))
      (clean-fuel-deployed ?u)
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Rollout electric induction in initially grid-ready unit
  (:action rollout-electric-init-ready
    :parameters (?u - unit ?p - period ?c1 ?c2 - coverage-tier ?b1 ?b2 - budget-level)
    :precondition (and
      (grid-init-ready ?u)
      (active-period ?p)
      (slot-open ?u ?p)
      (cov-tier ?u ?c1)
      (next-cov ?c1 ?c2)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (not (cov-tier ?u ?c1))
      (cov-tier ?u ?c2)
      (not (slot-open ?u ?p))
      (clean-fuel-deployed ?u)
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Rollout electric induction after grid built in a STRICTLY earlier period
  (:action rollout-electric-after-infra
    :parameters (?u - unit ?p_infra ?p - period ?c1 ?c2 - coverage-tier ?b1 ?b2 - budget-level)
    :precondition (and
      (grid-infra-built-at ?u ?p_infra)
      (strictly-before ?p_infra ?p)
      (active-period ?p)
      (slot-open ?u ?p)
      (cov-tier ?u ?c1)
      (next-cov ?c1 ?c2)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (not (cov-tier ?u ?c1))
      (cov-tier ?u ?c2)
      (not (slot-open ?u ?p))
      (clean-fuel-deployed ?u)
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Rollout improved biomass (no LPG/grid infrastructure prerequisite)
  (:action rollout-improved-biomass
    :parameters (?u - unit ?p - period ?c1 ?c2 - coverage-tier ?b1 ?b2 - budget-level)
    :precondition (and
      (active-period ?p)
      (slot-open ?u ?p)
      (cov-tier ?u ?c1)
      (next-cov ?c1 ?c2)
      (budget-level ?b1)
      (next-budget ?b1 ?b2)
    )
    :effect (and
      (not (cov-tier ?u ?c1))
      (cov-tier ?u ?c2)
      (not (slot-open ?u ?p))
      (not (budget-level ?b1))
      (budget-level ?b2)
    )
  )

  ;; Advance sequentially between 5-year periods
  (:action advance-period
    :parameters (?p1 ?p2 - period)
    :precondition (and
      (active-period ?p1)
      (next-period ?p1 ?p2)
    )
    :effect (and
      (not (active-period ?p1))
      (active-period ?p2)
    )
  )
)

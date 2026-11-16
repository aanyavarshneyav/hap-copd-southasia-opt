(define (problem ind-npl-clean-cooking-2026-2040)
  (:domain south-asia-clean-cooking)

  (:objects
    ind_urban ind_rural npl_urban npl_rural - unit
    p1 p2 p3 - period
    c0 c1 c2 - coverage-tier
    b0 b1 b2 b3 b4 b5 b6 b7 b8 - budget-level
  )

  (:init
    ;; Initial infrastructure readiness from baseline_unit_state
    (lpg-init-ready ind_urban)
    (grid-init-ready ind_urban)
    (lpg-init-ready npl_urban)
    (grid-init-ready npl_urban)

    ;; Initial zero additional coverage above 2040 BAU
    (cov-tier ind_urban c0)
    (cov-tier ind_rural c0)
    (cov-tier npl_urban c0)
    (cov-tier npl_rural c0)

    ;; Linear coverage tier chain (each step = +0.10 coverage increment)
    (next-cov c0 c1)
    (next-cov c1 c2)

    ;; Three 5-year periods (P1: 2026-2030, P2: 2031-2035, P3: 2036-2040)
    (active-period p1)
    (next-period p1 p2)
    (next-period p2 p3)
    (strictly-before p1 p2)
    (strictly-before p1 p3)
    (strictly-before p2 p3)

    ;; One cooking rollout slot per unit per period (enforces adoption cap)
    (slot-open ind_urban p1)
    (slot-open ind_urban p2)
    (slot-open ind_urban p3)
    (slot-open ind_rural p1)
    (slot-open ind_rural p2)
    (slot-open ind_rural p3)
    (slot-open npl_urban p1)
    (slot-open npl_urban p2)
    (slot-open npl_urban p3)
    (slot-open npl_rural p1)
    (slot-open npl_rural p2)
    (slot-open npl_rural p3)

    ;; Discrete budget levels (8 action units)
    (budget-level b8)
    (next-budget b8 b7)
    (next-budget b7 b6)
    (next-budget b6 b5)
    (next-budget b5 b4)
    (next-budget b4 b3)
    (next-budget b3 b2)
    (next-budget b2 b1)
    (next-budget b1 b0)
  )

  (:goal (and
    (cov-tier ind_urban c1)
    (cov-tier ind_rural c2)
    (cov-tier npl_urban c1)
    (cov-tier npl_rural c2)
    (clean-fuel-deployed ind_urban)
    (clean-fuel-deployed ind_rural)
    (clean-fuel-deployed npl_urban)
    (clean-fuel-deployed npl_rural)
  ))
)

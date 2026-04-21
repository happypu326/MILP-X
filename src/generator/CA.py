from __future__ import annotations
from pathlib import Path
import numpy as np
from .base import BaseGenerator


def generate_cauctions(random, filename, n_items=100, n_bids=500, min_value=1, max_value=100,
                       value_deviation=0.5, add_item_prob=0.7, max_n_sub_bids=5,
                       additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                       integers=False, warnings=False):
    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)

    # item compatibilities
    compats = np.triu(random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i+1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(f" +1 x{i+1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i+1}")


class CombinatorialAuctionsGenerator(BaseGenerator):
    problem_code = "CA"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        n_items: int | None = None,
        n_bids: int | None = None,
        min_value: int = 1,
        max_value: int = 100,
        value_deviation: float = 0.5,
        add_item_prob: float = 0.7,
        max_n_sub_bids: int = 5,
        additivity: float = 0.2,
        budget_factor: float = 1.5,
        resale_factor: float = 0.5,
        integers: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "very_easy":
            self.n_items = n_items if n_items is not None else 100
            self.n_bids = n_bids if n_bids is not None else 500
        elif difficulty == "easy":
            self.n_items = n_items if n_items is not None else 200
            self.n_bids = n_bids if n_bids is not None else 1000
        elif difficulty == "medium":
            self.n_items = n_items if n_items is not None else 300
            self.n_bids = n_bids if n_bids is not None else 1500
        elif difficulty == "very-hard":
            self.n_items = n_items if n_items is not None else 2000
            self.n_bids = n_bids if n_bids is not None else 4000
        elif difficulty == "very-hard2":
            self.n_items = n_items if n_items is not None else 4000
            self.n_bids = n_bids if n_bids is not None else 8000
        else:
            self.n_items = n_items if n_items is not None else 200
            self.n_bids = n_bids if n_bids is not None else 1000
        self.min_value = min_value
        self.max_value = max_value
        self.value_deviation = value_deviation
        self.add_item_prob = add_item_prob
        self.max_n_sub_bids = max_n_sub_bids
        self.additivity = additivity
        self.budget_factor = budget_factor
        self.resale_factor = resale_factor
        self.integers = integers
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        return {"seed": np.random.randint(2**31)}

    def _write_lp(self, instance, filepath: Path) -> None:
        rng = np.random.RandomState(instance["seed"])
        generate_cauctions(
            rng,
            filepath,
            n_items=self.n_items,
            n_bids=self.n_bids,
            min_value=self.min_value,
            max_value=self.max_value,
            value_deviation=self.value_deviation,
            add_item_prob=self.add_item_prob,
            max_n_sub_bids=self.max_n_sub_bids,
            additivity=self.additivity,
            budget_factor=self.budget_factor,
            resale_factor=self.resale_factor,
            integers=self.integers,
        )

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"ca_{self.n_items}i_{self.n_bids}b_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[CA] Generating instance {idx+1}: {filepath}")
        return filepath


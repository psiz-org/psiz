###############################
Hierarchical Variational Inference
###############################

:Author: Brett D. Roads

This page gives a high-level map of the hierarchical variational inference (VI)
stack in PsiZ and shows the main construction options.

When to use hierarchical VI
===========================

Use hierarchical VI when embeddings should share statistical strength across
levels (for example: global -> group -> item), while preserving uncertainty at
each level.

At a high level, PsiZ composes:

1. A root prior embedding.
2. One non-centered variational layer per non-root level.
3. KL terms that regularize each level relative to its parent level.

Construction options
====================

PsiZ exposes three progressive-disclosure entry points:

1. Function wrapper (fast start):

.. code-block:: python

    embedding = psiz.keras.layers.build_hierarchical_vi_embedding(
        n_stimuli=n_stimuli,
        n_dim=n_dim,
        hierarchy=hierarchy,
        membership=membership,
        posterior_factory=psiz.keras.layers.NonCenteredPosteriorFactory(),
        n_sample_train=n_sample_train,
    )

2. Standard builder (configurable defaults):

.. code-block:: python

    builder = psiz.keras.layers.HierarchicalVIEmbeddingBuilder(
        hierarchy=hierarchy,
        posterior_factory=psiz.keras.layers.NonCenteredPosteriorFactory(),
        kl_policy=psiz.keras.layers.KLWeightingPolicy.PER_SAMPLE_PER_BRANCH,
        scale_policy=psiz.keras.layers.ScaleInitializationPolicy.GEOMETRIC_DECAY,
    )
    embedding = builder.build(
        n_stimuli=n_stimuli,
        n_dim=n_dim,
        membership=membership,
        n_sample_train=n_sample_train,
    )

3. Advanced builder with hooks (expert customization):

.. code-block:: python

    hooks = psiz.keras.layers.PretrainedNonCenteredFactoryHooks()
    builder = psiz.keras.layers.AdvancedHierarchicalVIEmbeddingBuilder(
        hierarchy=hierarchy,
        posterior_factory=psiz.keras.layers.NonCenteredPosteriorFactory(),
        hooks=hooks,
    )
    embedding = builder.build(
        n_stimuli=n_stimuli,
        n_dim=n_dim,
        membership=membership,
        n_sample_train=n_sample_train,
    )

Core specification objects
==========================

HierarchySpec
-------------

Defines the ordered hierarchy and masking behavior:

- levels: list of HierarchyLevelSpec objects.
- mask_zero: whether index 0 is reserved as a masked token.

HierarchyLevelSpec
------------------

Defines one level in the hierarchy:

- role: semantic name of the level.
- membership_key: optional dataframe key for membership resolution.
- loc_trainable and scale_trainable: parameter trainability controls.
- initialization: default, pretrained, or pretrained_point_estimate.
- metadata: optional payload for level-specific initialization details.

MembershipInput
---------------

Supplies hierarchy memberships through one of these sources:

- memberships: precomputed integer matrix of shape [n_stimuli, n_levels].
- df_stimuli: dataframe input used by resolver-based modes.
- resolver_name: optional label for resolver strategy.

Policy enums
============

KL weighting
------------

- KLWeightingPolicy.PER_SAMPLE_PER_BRANCH
- KLWeightingPolicy.PER_SAMPLE
- KLWeightingPolicy.PER_SAMPLE_PER_LEVEL
- KLWeightingPolicy.CUSTOM

Scale initialization
--------------------

- ScaleInitializationPolicy.GEOMETRIC_DECAY
- ScaleInitializationPolicy.CONSTANT
- ScaleInitializationPolicy.CUSTOM

Membership source
-----------------

- MembershipSourcePolicy.PRECOMPUTED
- MembershipSourcePolicy.DATAFRAME_RESOLVER
- MembershipSourcePolicy.CUSTOM
- MembershipSourcePolicy.STRICT_PRECOMPUTED_OVERRIDES_RESOLVER

Parent map
----------

- ParentMapPolicy.MINIMAL_FIRST_OCCURRENCE
- ParentMapPolicy.FULL_IDENTITY
- ParentMapPolicy.CUSTOM

Key code components
===================

Public builder and specs
------------------------

- psiz.keras.layers.build_hierarchical_vi_embedding
- psiz.keras.layers.HierarchicalVIEmbeddingBuilder
- psiz.keras.layers.AdvancedHierarchicalVIEmbeddingBuilder
- psiz.keras.layers.HierarchySpec
- psiz.keras.layers.HierarchyLevelSpec
- psiz.keras.layers.MembershipInput

Variational primitives
----------------------

- psiz.keras.layers.EmbeddingNonCenteredVariational
- psiz.keras.layers.EmbeddingNonCenteredNormalDiag
- psiz.keras.layers.PosteriorFactory
- psiz.keras.layers.NonCenteredPosteriorFactory

Optional pretrained hooks
-------------------------

- psiz.keras.layers.PretrainedNonCenteredFactoryHooks

This hook enables pretrained and pretrained_point_estimate initialization modes
for non-centered posterior factories in advanced builder workflows.

Notes on initialization modes
=============================

The standard builder intentionally raises NotImplementedError for pretrained
initialization modes unless a project-specific hook strategy is provided.
This keeps default behavior explicit and avoids hidden assumptions about
external checkpoint formats.

For advanced workflows, use AdvancedHierarchicalVIEmbeddingBuilder with
PretrainedNonCenteredFactoryHooks (or a custom hooks implementation).

For a complete end-to-end training example on birds-16 using the hierarchical
builder stack, see :doc:`Introduction to Hierarchical Variational Inference <birds16_hierarchical_vi>`.

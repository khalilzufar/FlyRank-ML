# Analytics boundary

The site includes a small first-party interaction counter for the case explorer. It stores only aggregate event counts in the visitor's own browser `localStorage`; it does not send data to a server, set a third-party cookie, or collect an identifier. This makes the feature testable without pretending that an external analytics account exists.

For a real analytics dashboard later, the owner would choose a provider, create the property, add the provider's measurement identifier, verify the privacy notice, and capture a screenshot of the live dashboard. No such account or identifier was available for this build, so no fabricated screenshot or claim is included.

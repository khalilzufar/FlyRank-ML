# DNS walkthrough

DNS is the directory that lets a human-friendly name resolve to the host serving a site.

1. A browser asks its configured recursive resolver for the address.
2. If the resolver has no cached answer, it follows the DNS hierarchy to the authoritative nameserver for the domain.
3. The nameserver returns a record such as an `A` record (an IP address) or a `CNAME` record (an alias to another hostname).
4. The resolver returns the answer with a TTL, and the browser connects to the hosting provider over HTTPS.

A `CNAME` is useful for a subdomain such as `www.example.com` because it points that name at the host's canonical hostname. It is not the same thing as a redirect and it cannot normally be used at the root/apex of every DNS provider. The provider then verifies the record before issuing or attaching a TLS certificate.

This project currently uses the accepted free GitHub Pages URL:
`https://khalilzufar.github.io/FlyRank-ML/ai/`.

No custom domain was purchased or configured. If a custom domain is added later, the provider's DNS instructions must be followed exactly and verified with `dig`, the provider dashboard, and an HTTPS request.

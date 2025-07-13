interface DigitalOceanAccount {
    uuid: string;
    email: string;
    email_verified: boolean;
    status: string;
    status_message: string;
}
interface ApiValidationResult {
    valid: boolean;
    account?: DigitalOceanAccount;
    error?: string;
}
declare class DigitalOceanService {
    private apiClient;
    constructor(apiKey: string);
    validateApiKey(): Promise<ApiValidationResult>;
    static createFromConfig(): Promise<DigitalOceanService | null>;
}
declare const digitalOceanService: {
    create: (apiKey: string) => DigitalOceanService;
    createFromConfig: () => Promise<DigitalOceanService | null>;
};

export { type ApiValidationResult, type DigitalOceanAccount, DigitalOceanService, digitalOceanService };
